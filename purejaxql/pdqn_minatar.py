"""
Example script with PLANNING_DELAY_UPDATES using jax.lax.cond
"""
import os
import copy 
import time
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import Any

import chex
import optax
import flax.linen as nn
from flax.training.train_state import TrainState
from gymnax.wrappers.purerl import FlattenObservationWrapper, LogWrapper
import hydra
from omegaconf import OmegaConf
import gymnax
import wandb


class CNN(nn.Module):
    norm_type: str = "layer_norm"

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool):
        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        elif self.norm_type == "batch_norm":
            normalize = lambda x: nn.BatchNorm(use_running_average=not train)(x)
        else:
            normalize = lambda x: x

        x = nn.Conv(
            16,
            kernel_size=(3, 3),
            strides=1,
            padding="VALID",
            kernel_init=nn.initializers.he_normal(),
        )(x)
        x = normalize(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(128, kernel_init=nn.initializers.he_normal())(x)
        x = normalize(x)
        x = nn.relu(x)
        return x


class QNetwork(nn.Module):
    action_dim: int
    norm_type: str = "layer_norm"
    norm_input: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool):
        if self.norm_input:
            x = nn.BatchNorm(use_running_average=not train)(x)
        else:
            x_dummy = nn.BatchNorm(use_running_average=not train)(x)
            x = x / 255.0
        x = CNN(norm_type=self.norm_type)(x, train)
        x = nn.Dense(self.action_dim)(x)
        return x


class WorldModel(nn.Module):
    action_dim: int
    norm_type: str = "layer_norm"
    norm_input: bool = False

    @nn.compact
    def __call__(self, obs: jnp.ndarray, action: jnp.ndarray, train: bool):
        if self.norm_input:
            obs = nn.BatchNorm(use_running_average=not train)(obs)
        else:
            obs_dummy = nn.BatchNorm(use_running_average=not train)(obs)
            obs = obs / 255.0

        x = CNN(norm_type=self.norm_type)(obs, train)
        action_oh = jax.nn.one_hot(action, self.action_dim)
        x = jnp.concatenate([x, action_oh], axis=-1)

        obs_shape = obs.shape[1:]
        next_obs_dim = np.prod(obs_shape)

        # Predict next observation
        next_obs_pred = nn.Dense(next_obs_dim)(x)
        next_obs_pred = next_obs_pred.reshape((obs.shape[0], *obs_shape))

        # Predict reward
        reward_pred = nn.Dense(1)(x).squeeze(-1)

        # --- ADD THIS for done prediction: single scalar (logit) ---
        done_pred = nn.Dense(1)(x).squeeze(-1)

        # Now return 3 outputs:
        return (next_obs_pred, reward_pred, done_pred)



@chex.dataclass(frozen=True)
class Transition:
    obs: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array
    next_obs: chex.Array
    q_val: chex.Array


class CustomTrainState(TrainState):
    batch_stats: Any
    timesteps: int = 0
    n_updates: int = 0
    grad_steps: int = 0


def make_train(config):

    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["NUM_UPDATES_DECAY"] = (
        config["TOTAL_TIMESTEPS_DECAY"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )

    assert (config["NUM_STEPS"] * config["NUM_ENVS"]) % config["NUM_MINIBATCHES"] == 0,\
        "NUM_MINIBATCHES must divide NUM_STEPS*NUM_ENVS"

    env, env_params = gymnax.make(config["ENV_NAME"])
    env = LogWrapper(env)
    config["TEST_NUM_STEPS"] = env_params.max_steps_in_episode

    # Vectorized reset & step
    vmap_reset = lambda n_envs: lambda rng: jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(rng, n_envs), env_params
    )
    vmap_step = lambda n_envs: lambda rng, env_state, action: jax.vmap(
        env.step, in_axes=(0, 0, 0, None)
    )(jax.random.split(rng, n_envs), env_state, action, env_params)

    def eps_greedy_exploration(rng, q_vals, eps):
        rng_a, rng_e = jax.random.split(rng)
        greedy_actions = jnp.argmax(q_vals, axis=-1)
        chosen_actions = jnp.where(
            jax.random.uniform(rng_e, greedy_actions.shape) < eps,
            jax.random.randint(
                rng_a, shape=greedy_actions.shape, minval=0, maxval=q_vals.shape[-1]
            ),
            greedy_actions,
        )
        return chosen_actions

    def train(rng):
        """
        Main training loop, vmap-ed over seeds later.
        """

        original_rng = rng[0]

        # Epsilon schedules
        eps_scheduler = optax.linear_schedule(
            config["EPS_START"],
            config["EPS_FINISH"],
            config["EPS_DECAY"] * config["NUM_UPDATES_DECAY"],
        )
        eps_scheduler_plan = optax.linear_schedule(
            config["EPS_START_PLAN"],
            config["EPS_FINISH_PLAN"],
            config["EPS_DECAY_PLAN"] * config["NUM_UPDATES_DECAY"],
        )

        lr_scheduler = optax.linear_schedule(
            init_value=config["LR"],
            end_value=1e-20,
            transition_steps=(config["NUM_UPDATES_DECAY"])
            * config["NUM_MINIBATCHES"]
            * config["NUM_EPOCHS"],
        )
        lr = lr_scheduler if config.get("LR_LINEAR_DECAY", False) else config["LR"]

        # Create Q-network
        network = QNetwork(
            action_dim=env.action_space(env_params).n,
            norm_type=config["NORM_TYPE"],
            norm_input=config.get("NORM_INPUT", False),
        )

        def create_agent(rng):
            init_x = jnp.zeros((1, *env.observation_space(env_params).shape))
            network_vars = network.init(rng, init_x, train=False)
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.radam(learning_rate=lr),
            )
            return CustomTrainState.create(
                apply_fn=network.apply,
                params=network_vars["params"],
                batch_stats=network_vars["batch_stats"],
                tx=tx,
            )

        # Create world model
        def create_model(rng):
            wm = WorldModel(
                action_dim=env.action_space(env_params).n,
                norm_type=config["NORM_TYPE"],
                norm_input=config.get("NORM_INPUT", False),
            )
            init_x = jnp.zeros((1, *env.observation_space(env_params).shape))
            init_action = jnp.zeros((1,), dtype=jnp.int32)
            wm_vars = wm.init(rng, init_x, init_action, train=False)

            tx_model = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.radam(learning_rate=lr),
            )
            return wm, CustomTrainState.create(
                apply_fn=wm.apply,
                params=wm_vars["params"],
                batch_stats=wm_vars["batch_stats"],
                tx=tx_model,
            )

        # Initialize
        rng, rng_agent = jax.random.split(rng)
        train_state = create_agent(rng_agent)

        rng, rng_wm = jax.random.split(rng)
        world_model, model_state = create_model(rng_wm)

        # Optional testing function
        def get_test_metrics(train_state, rng):
            if not config.get("TEST_DURING_TRAINING", False):
                return None

            def _env_step(carry, _):
                env_state, last_obs, rng = carry
                rng, _rng = jax.random.split(rng)
                q_vals = network.apply(
                    {
                        "params": train_state.params,
                        "batch_stats": train_state.batch_stats,
                    },
                    last_obs,
                    train=False,
                )
                action = jnp.argmax(q_vals, axis=-1)
                new_obs, new_env_state, reward, done, info = vmap_step(
                    config["TEST_NUM_ENVS"]
                )(_rng, env_state, action)
                return (new_env_state, new_obs, rng), info

            rng, rng_init = jax.random.split(rng)
            init_obs, env_state = vmap_reset(config["TEST_NUM_ENVS"])(rng_init)

            (_, _, _), infos = jax.lax.scan(
                _env_step, (env_state, init_obs, rng_init), None, config["TEST_NUM_STEPS"]
            )
            done_infos = jax.tree_map(
                lambda x: jnp.nanmean(
                    jnp.where(infos["returned_episode"], x, jnp.nan)
                ),
                infos,
            )
            return done_infos

        # ------------------------------------------------
        #                MAIN UPDATE FUNCTION
        # ------------------------------------------------
        def _update_step(runner_state, _):
            """
            Single training iteration inside jax.lax.scan.
            runner_state = (train_state, expl_state, test_metrics, model_state, rng)
            """
            train_state, expl_state, test_metrics, model_state, rng = runner_state

            # 1) Collect transitions from the real environment
            def _step_env(carry, _):
                last_obs, env_state, rng_inner = carry
                rng_inner, rng_a, rng_s = jax.random.split(rng_inner, 3)

                q_vals = network.apply(
                    {
                        "params": train_state.params,
                        "batch_stats": train_state.batch_stats,
                    },
                    last_obs,
                    train=False,
                )
                eps_val = jnp.full(config["NUM_ENVS"], eps_scheduler(train_state.n_updates))
                rngs_action = jax.random.split(rng_a, config["NUM_ENVS"])
                new_action = jax.vmap(eps_greedy_exploration)(rngs_action, q_vals, eps_val)

                new_obs, new_env_state, reward, new_done, info = vmap_step(config["NUM_ENVS"])(
                    rng_s, env_state, new_action
                )

                transition = Transition(
                    obs=last_obs,
                    action=new_action,
                    reward=config.get("REW_SCALE", 1) * reward,
                    done=new_done,
                    next_obs=new_obs,
                    q_val=q_vals,
                )
                return (new_obs, new_env_state, rng_inner), (transition, info)

            rng, rng_expl = jax.random.split(rng)
            (final_obs, final_env_state, _), (transitions, infos) = jax.lax.scan(
                _step_env, (*expl_state, rng_expl), None, config["NUM_STEPS"]
            )
            expl_state = (final_obs, final_env_state)

            train_state = train_state.replace(
                timesteps=train_state.timesteps + config["NUM_STEPS"] * config["NUM_ENVS"]
            )

            # 2) Compute lambda-returns for Q updates
            last_q = network.apply(
                {"params": train_state.params, "batch_stats": train_state.batch_stats},
                transitions.next_obs[-1],
                train=False,
            )
            last_q = jnp.max(last_q, axis=-1)

            def _get_target(carry, transition):
                lambda_returns, next_q = carry
                target_bootstrap = transition.reward + config["GAMMA"] * (1 - transition.done) * next_q
                delta = lambda_returns - next_q
                lambda_returns = target_bootstrap + config["GAMMA"] * config["LAMBDA"] * delta
                lambda_returns = (
                    (1 - transition.done) * lambda_returns + transition.done * transition.reward
                )
                next_q = jnp.max(transition.q_val, axis=-1)
                return (lambda_returns, next_q), lambda_returns

            # final lambda-returns
            last_q = last_q * (1 - transitions.done[-1])
            lambda_returns = transitions.reward[-1] + config["GAMMA"] * last_q
            _, targets = jax.lax.scan(
                _get_target,
                (lambda_returns, last_q),
                jax.tree_map(lambda x: x[:-1], transitions),
                reverse=True,
            )
            lambda_targets = jnp.concatenate((targets, lambda_returns[np.newaxis]))

            # 3) Q-network update (real transitions)
            def _learn_epoch(carry, _):
                train_state_local, rng_local = carry

                def _learn_phase(carry2, mb_target):
                    train_state_inner, rng_inner = carry2
                    mb, targ = mb_target

                    def _loss_fn(params):
                        q_vals_local, updates_local = network.apply(
                            {"params": params, "batch_stats": train_state_inner.batch_stats},
                            mb.obs,
                            train=True,
                            mutable=["batch_stats"],
                        )
                        chosen_qvals = jnp.take_along_axis(
                            q_vals_local,
                            jnp.expand_dims(mb.action, axis=-1),
                            axis=-1,
                        ).squeeze(axis=-1)
                        loss_val = 0.5 * jnp.square(chosen_qvals - targ).mean()
                        return loss_val, (updates_local, chosen_qvals)

                    (loss_val, (updates_local, chosen_qvals)), grads = jax.value_and_grad(
                        _loss_fn, has_aux=True
                    )(train_state_inner.params)

                    train_state_inner = train_state_inner.apply_gradients(grads=grads)
                    train_state_inner = train_state_inner.replace(
                        grad_steps=train_state_inner.grad_steps + 1,
                        batch_stats=updates_local["batch_stats"],
                    )
                    return (train_state_inner, rng_inner), (loss_val, chosen_qvals)

                # Shuffle transitions into minibatches
                def preprocess_transition(x, rng_pp):
                    x = x.reshape(-1, *x.shape[2:])  # flatten steps + envs
                    x = jax.random.permutation(rng_pp, x)
                    x = x.reshape(config["NUM_MINIBATCHES"], -1, *x.shape[1:])
                    return x

                rng_local, rng_shuf = jax.random.split(rng_local)
                minibatches = jax.tree_map(lambda x: preprocess_transition(x, rng_shuf), transitions)
                targs = jax.tree_map(lambda x: preprocess_transition(x, rng_shuf), lambda_targets)

                # Now iterate over each minibatch:
                rng_local, rng_epoch = jax.random.split(rng_local)
                (train_state_local, rng_local), (loss_arr, qvals_arr) = jax.lax.scan(
                    _learn_phase, (train_state_local, rng_local), (minibatches, targs)
                )

                return (train_state_local, rng_local), (loss_arr, qvals_arr)

            rng, rng_qupdate = jax.random.split(rng)
            (train_state, rng_qupdate), (loss_arr, qvals_arr) = jax.lax.scan(
                _learn_epoch, (train_state, rng_qupdate), None, config["NUM_EPOCHS"]
            )

            train_state = train_state.replace(n_updates=train_state.n_updates + 1)

            metrics = {
                "env_step": train_state.timesteps,
                "update_steps": train_state.n_updates,
                "env_frame": train_state.timesteps * env.observation_space(env_params).shape[-1],
                "grad_steps": train_state.grad_steps,
                "td_loss": loss_arr.mean(),
                "qvals": qvals_arr.mean(),
            }

            # 4) Logging env info
            metrics.update({k: v.mean() for k, v in infos.items()})

            # 5) Optional test metrics
            if config.get("TEST_DURING_TRAINING", False):
                rng, rng_test = jax.random.split(rng)
                def get_tm(_):
                    return get_test_metrics(train_state, rng_test)
                test_metrics = jax.lax.cond(
                    train_state.n_updates % int(config["NUM_UPDATES"] * config["TEST_INTERVAL"]) == 0,
                    get_tm,
                    lambda _: test_metrics,
                    operand=None
                )
                if test_metrics is not None:
                    metrics.update({f"test_{k}": v for k, v in test_metrics.items()})

            # 6) Update world model
            def model_loss(params, batch_stats, obs, action, next_obs, reward, done):
                # Now we expect three outputs: (next_obs_pred, reward_pred, done_pred)
                (next_obs_pred, reward_pred, done_pred), updates_local = world_model.apply(
                    {"params": params, "batch_stats": batch_stats},
                    obs,
                    action,
                    train=True,
                    mutable=["batch_stats"],
                )
                # For simplicity, use an MSE for done. 
                # (If you prefer a binary cross-entropy, you can use e.g. 
                #   optax.sigmoid_binary_cross_entropy(logits=done_pred, labels=done).mean()
                # depending on how you want 'done_pred' to represent the probability.)
                loss_val = ((next_obs_pred - next_obs) ** 2).mean() \
                        + ((reward_pred - reward) ** 2).mean() \
                        + ((done_pred - done) ** 2).mean()
                return loss_val, updates_local


            def preprocess_for_model(x):
                return x.reshape(-1, *x.shape[2:])  # flatten steps + envs

            obs_batch = preprocess_for_model(transitions.obs)
            act_batch = preprocess_for_model(transitions.action)
            next_obs_batch = preprocess_for_model(transitions.next_obs)
            rew_batch = preprocess_for_model(transitions.reward)
            done_batch = preprocess_for_model(transitions.done)  # <--- ADD THIS

            (m_loss, wm_updates), wm_grads = jax.value_and_grad(model_loss, has_aux=True)(
                model_state.params,
                model_state.batch_stats,
                obs_batch,
                act_batch,
                next_obs_batch,
                rew_batch,
                done_batch,
            )
            model_state = model_state.apply_gradients(grads=wm_grads)
            model_state = model_state.replace(batch_stats=wm_updates["batch_stats"])
            metrics["model_loss"] = m_loss

            # ------------------------------------------------
            # 7) PLANNING PHASE, BUT DELAYED
            # ------------------------------------------------
            # <<< CHANGED / ADDED >>>
            # We'll define two sub-functions for jax.lax.cond:

            def _planning_phase(carry):
                """
                carry = (train_state, model_state, rng, s_0, rng_planning)
                Returns ((train_state, model_state, rng), plan_metrics)
                """
                train_state_local, model_state_local, rng_local, s0, rng_planning = carry

                n_planning = config.get("PLANNING_STEPS", 0)

                def _step_model(carry2, _):
                    """
                    Single model-based step (like environment).
                    carry2 = (obs, model_state, rng2)
                    """
                    last_obs_mb, mod_st, rng2 = carry2
                    rng2, rng_a, rng_s = jax.random.split(rng2, 3)

                    q_vals_mb = network.apply(
                        {"params": train_state_local.params, "batch_stats": train_state_local.batch_stats},
                        last_obs_mb,
                        train=False,
                    )
                    eps_plan_val = jnp.full(last_obs_mb.shape[0], eps_scheduler_plan(train_state_local.n_updates))
                    rngs_action_mb = jax.random.split(rng_a, last_obs_mb.shape[0])
                    new_action_mb = jax.vmap(eps_greedy_exploration)(rngs_action_mb, q_vals_mb, eps_plan_val)

                    (next_obs_pred, reward_pred, done_pred), _ = world_model.apply(
                        {"params": model_state_local.params, "batch_stats": model_state_local.batch_stats},
                        last_obs_mb,
                        new_action_mb,
                        train=False,
                        mutable=[],
                    )

                    # For example, turn it into a 0/1 mask via sigmoid:
                    done_prob_mb = jax.nn.sigmoid(done_pred)
                    new_done_mb = jnp.where(done_prob_mb >= 0.5, 1.0, 0.0).astype(jnp.float32)


                    transition_mb = Transition(
                        obs=last_obs_mb,
                        action=new_action_mb,
                        reward=config.get("REW_SCALE", 1) * reward_pred,
                        done=new_done_mb,
                        next_obs=next_obs_pred,
                        q_val=q_vals_mb,
                    )
                    return (next_obs_pred, mod_st, rng2), (transition_mb, {})

                # Roll out the model for n_planning steps
                (final_obs_mb, _, _), (model_transitions, _) = jax.lax.scan(
                    _step_model, (s0, model_state_local, rng_planning), None, length=n_planning
                )

                # Now do Q-updates from these model transitions
                def _learn_model_phase(carry3, transition_mb):
                    train_state_m, rng_m = carry3
                    # 1-step target
                    next_q_mb = network.apply(
                        {
                            "params": train_state_m.params,
                            "batch_stats": train_state_m.batch_stats,
                        },
                        transition_mb.next_obs,
                        train=False,
                    )
                    max_next_q_mb = jnp.max(next_q_mb, axis=-1)
                    target_mb = transition_mb.reward + config["GAMMA"] * (1 - transition_mb.done) * max_next_q_mb

                    def _loss_fn(params):
                        q_vals_planning, upd_plan = network.apply(
                            {"params": params, "batch_stats": train_state_m.batch_stats},
                            transition_mb.obs,
                            train=True,
                            mutable=["batch_stats"],
                        )
                        chosen_q_planning = jnp.take_along_axis(
                            q_vals_planning,
                            jnp.expand_dims(transition_mb.action, axis=-1),
                            axis=-1
                        ).squeeze(-1)
                        loss_planning = 0.5 * jnp.square(
                            chosen_q_planning - jax.lax.stop_gradient(target_mb)
                        ).mean()
                        return loss_planning, (upd_plan, chosen_q_planning)

                    (loss_pl, (upd_plan, chosen_q_planning)), grads_pl = jax.value_and_grad(
                        _loss_fn, has_aux=True
                    )(train_state_m.params)

                    train_state_m = train_state_m.apply_gradients(grads=grads_pl)
                    train_state_m = train_state_m.replace(
                        grad_steps=train_state_m.grad_steps + 1,
                        batch_stats=upd_plan["batch_stats"],
                    )
                    return (train_state_m, rng_m), (loss_pl, chosen_q_planning)

                rng_local, rng_model_update = jax.random.split(rng_local)
                (train_state_local, _), (model_loss_arr, model_qvals_arr) = jax.lax.scan(
                    _learn_model_phase, (train_state_local, rng_model_update), model_transitions
                )

                plan_metrics = {
                    "planning_loss": model_loss_arr.mean(),
                    "planning_qvals": model_qvals_arr.mean(),
                }
                return (train_state_local, model_state_local, rng_local), plan_metrics

            def _skip_planning_phase(carry):
                """
                carry = (train_state, model_state, rng, s_0, rng_planning)
                Return the same state, plus dummy metrics
                """
                (train_state_local, model_state_local, rng_local, _, _) = carry
                plan_metrics = {"planning_loss": 0.0, "planning_qvals": 0.0}
                return (train_state_local, model_state_local, rng_local), plan_metrics

            # The carry includes s_0 = transitions.next_obs[-1],
            # so we can start planning from the last real next_obs.
            s_0 = transitions.next_obs[-1]

            # We'll combine conditions: only plan if we've reached the delay threshold
            # AND PLANNING_STEPS > 0
            do_plan_condition = jnp.logical_and(
                jnp.array(train_state.n_updates) >= config["PLANNING_DELAY_UPDATES"],
                jnp.array(config["PLANNING_STEPS"]) > 0
            )

            # Use lax.cond to either do planning or skip it
            (train_state, model_state, rng), plan_metrics = jax.lax.cond(
                do_plan_condition,
                _planning_phase,
                _skip_planning_phase,
                (train_state, model_state, rng, s_0, rng),
            )

            metrics.update(plan_metrics)

            # 8) Log to wandb if enabled
            if config["WANDB_MODE"] != "disabled":
                def callback(m, orig_rng):
                    wandb.log(m, step=m["update_steps"])
                jax.debug.callback(callback, metrics, original_rng)

            runner_state_out = (train_state, expl_state, test_metrics, model_state, rng)
            return runner_state_out, metrics

        # ------------------------------------------------
        #            END _update_step DEFINITION
        # ------------------------------------------------

        # Possibly get initial test metrics
        rng, rng_test_init = jax.random.split(rng)
        test_metrics_init = get_test_metrics(train_state, rng_test_init)

        # Reset environment for exploration
        rng, rng_reset = jax.random.split(rng)
        init_obs, init_env_state = vmap_reset(config["NUM_ENVS"])(rng_reset)
        expl_state = (init_obs, init_env_state)

        # Runner state
        runner_state = (train_state, expl_state, test_metrics_init, model_state, rng)

        # Main training loop using lax.scan
        runner_state, metrics_scan = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )

        return {"runner_state": runner_state, "metrics": metrics_scan}

    return train


def single_run(config):
    # Merge config dict
    config = {**config, **config["alg"]}
    print(config)

    alg_name = config.get("ALG_NAME", "pqn")
    env_name = config["ENV_NAME"]

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=[
            alg_name.upper(),
            env_name.upper(),
            f"jax_{jax.__version__}",
        ],
        name=f'{config["ALG_NAME"]}_{config["ENV_NAME"]}',
        config=config,
        mode=config["WANDB_MODE"],
    )

    rng = jax.random.PRNGKey(config["SEED"])

    t0 = time.time()
    rngs = jax.random.split(rng, config["NUM_SEEDS"])

    train_fn = make_train(config)
    train_vjit = jax.jit(jax.vmap(train_fn))
    outs = jax.block_until_ready(train_vjit(rngs))

    print(f"Took {time.time() - t0} seconds to complete.")

    if config.get("SAVE_PATH", None) is not None:
        from jaxmarl.wrappers.baselines import save_params

        model_state = outs["runner_state"][0]
        save_dir = os.path.join(config["SAVE_PATH"], env_name)
        os.makedirs(save_dir, exist_ok=True)

        OmegaConf.save(
            config,
            os.path.join(
                save_dir, f'{alg_name}_{env_name}_seed{config["SEED"]}_config.yaml'
            ),
        )

        for i, seed_rng in enumerate(rngs):
            params_i = jax.tree_map(lambda x: x[i], model_state.params)
            save_path = os.path.join(
                save_dir,
                f'{alg_name}_{env_name}_seed{config["SEED"]}_vmap{i}.safetensors',
            )
            save_params(params_i, save_path)


def tune(default_config):
    """
    Example of hyperparameter sweep with wandb.
    """
    default_config = {**default_config, **default_config["alg"]}
    print(default_config)

    alg_name = default_config.get("ALG_NAME", "pqn")
    env_name = default_config["ENV_NAME"]

    def wrapped_make_train():
        wandb.init(project=default_config["PROJECT"])
        config_sweep = copy.deepcopy(default_config)
        for k, v in dict(wandb.config).items():
            config_sweep[k] = v

        print("running experiment with params:", config_sweep)
        rng = jax.random.PRNGKey(config_sweep["SEED"])
        rngs = jax.random.split(rng, config_sweep["NUM_SEEDS"])
        train_vjit = jax.jit(jax.vmap(make_train(config_sweep)))
        _outs = jax.block_until_ready(train_vjit(rngs))

    sweep_config = {
        "name": f"{alg_name}_{env_name}",
        "method": "bayes",
        "metric": {
            "name": "test_returned_episode_returns",
            "goal": "maximize",
        },
        "parameters": {
            "PLANNING_STEPS": {
                "values": [2, 5, 10, 20]
            },
            "PLANNING_DELAY_UPDATES": {
                "values": [0, 100, 500, 1000, 5000]
            },
        },
    }

    wandb.login()
    sweep_id = wandb.sweep(
        sweep_config, 
        entity=default_config["ENTITY"], 
        project=default_config["PROJECT"]
    )
    wandb.agent(sweep_id, wrapped_make_train, count=1000)


@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(config):
    config = OmegaConf.to_container(config)
    print("Config:\n", OmegaConf.to_yaml(config))

    # <<< CHANGED / ADDED >>>
    # Suppose you add in your config something like:
    #   PLANNING_STEPS: 5
    #   PLANNING_DELAY_UPDATES: 100
    # so that planning only begins after 100 updates.

    if config["HYP_TUNE"]:
        tune(config)
    else:
        single_run(config)


if __name__ == "__main__":
    main()
