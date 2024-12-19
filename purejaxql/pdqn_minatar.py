"""
This script is like pqn_gymnax but the network uses a CNN.
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
            # dummy normalize input for global compatibility
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
        # If needed, normalize input similar to QNetwork
        if self.norm_input:
            obs = nn.BatchNorm(use_running_average=not train)(obs)
        else:
            obs_dummy = nn.BatchNorm(use_running_average=not train)(obs)
            obs = obs / 255.0

        # Use the same CNN backbone
        x = CNN(norm_type=self.norm_type)(obs, train)

        # Incorporate action information (one-hot encode)
        action_oh = jax.nn.one_hot(action, self.action_dim)
        x = jnp.concatenate([x, action_oh], axis=-1)

        # Infer the observation shape from the input 'obs'
        # obs.shape is (batch_size, *obs_dims)
        obs_shape = obs.shape[1:]
        next_obs_dim = np.prod(obs_shape)
        
        # Predict next_obs
        next_obs_pred = nn.Dense(next_obs_dim)(x)
        next_obs_pred = next_obs_pred.reshape((obs.shape[0], *obs_shape))

        # Predict reward
        reward_pred = nn.Dense(1)(x).squeeze(-1)

        return (next_obs_pred, reward_pred)


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

    assert (config["NUM_STEPS"] * config["NUM_ENVS"]) % config[
        "NUM_MINIBATCHES"
    ] == 0, "NUM_MINIBATCHES must divide NUM_STEPS*NUM_ENVS"

    env, env_params = gymnax.make(config["ENV_NAME"])
    env = LogWrapper(env)
    config["TEST_NUM_STEPS"] = env_params.max_steps_in_episode

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

        original_rng = rng[0]

        eps_scheduler = optax.linear_schedule(
            config["EPS_START"],
            config["EPS_FINISH"],
            (config["EPS_DECAY"]) * config["NUM_UPDATES_DECAY"],
        )

        lr_scheduler = optax.linear_schedule(
            init_value=config["LR"],
            end_value=1e-20,
            transition_steps=(config["NUM_UPDATES_DECAY"])
            * config["NUM_MINIBATCHES"]
            * config["NUM_EPOCHS"],
        )
        lr = lr_scheduler if config.get("LR_LINEAR_DECAY", False) else config["LR"]

        # INIT NETWORK AND OPTIMIZER
        network = QNetwork(
            action_dim=env.action_space(env_params).n,
            norm_type=config["NORM_TYPE"],
            norm_input=config.get("NORM_INPUT", False),
        )

        def create_agent(rng):
            init_x = jnp.zeros((1, *env.observation_space(env_params).shape))
            network_variables = network.init(rng, init_x, train=False)
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.radam(learning_rate=lr),
            )

            train_state = CustomTrainState.create(
                apply_fn=network.apply,
                params=network_variables["params"],
                batch_stats=network_variables["batch_stats"],
                tx=tx,
            )
            return train_state

        def create_model(rng):
            world_model = WorldModel(
                action_dim=env.action_space(env_params).n,
                norm_type=config["NORM_TYPE"],
                norm_input=config.get("NORM_INPUT", False),
            )

            init_x = jnp.zeros((1, *env.observation_space(env_params).shape))
            init_action = jnp.zeros((1,), dtype=jnp.int32)
            world_model_variables = world_model.init(rng, init_x, init_action, train=False)

            tx_model = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.radam(learning_rate=lr),
            )

            model_state = CustomTrainState.create(
                apply_fn=world_model.apply,
                params=world_model_variables["params"],
                batch_stats=world_model_variables["batch_stats"],
                tx=tx_model,
            )
            return world_model, model_state

        rng, _rng = jax.random.split(rng)
        train_state = create_agent(rng)

        rng, _rng = jax.random.split(rng)
        world_model, model_state = create_model(_rng)

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

            rng, _rng = jax.random.split(rng)
            init_obs, env_state = vmap_reset(config["TEST_NUM_ENVS"])(_rng)

            _, infos = jax.lax.scan(
                _env_step, (env_state, init_obs, _rng), None, config["TEST_NUM_STEPS"]
            )
            done_infos = jax.tree_map(
                lambda x: jnp.nanmean(
                    jnp.where(
                        infos["returned_episode"],
                        x,
                        jnp.nan,
                    )
                ),
                infos,
            )
            return done_infos


        def _update_step(runner_state, unused):
            # runner_state now includes model_state
            train_state, expl_state, test_metrics, model_state, rng = runner_state

            # SAMPLE PHASE
            def _step_env(carry, _):
                last_obs, env_state, rng = carry
                rng, rng_a, rng_s = jax.random.split(rng, 3)
                q_vals = network.apply(
                    {
                        "params": train_state.params,
                        "batch_stats": train_state.batch_stats,
                    },
                    last_obs,
                    train=False,
                )

                _rngs = jax.random.split(rng_a, config["NUM_ENVS"])
                eps = jnp.full(config["NUM_ENVS"], eps_scheduler(train_state.n_updates))
                new_action = jax.vmap(eps_greedy_exploration)(_rngs, q_vals, eps)

                new_obs, new_env_state, reward, new_done, info = vmap_step(
                    config["NUM_ENVS"]
                )(rng_s, env_state, new_action)

                transition = Transition(
                    obs=last_obs,
                    action=new_action,
                    reward=config.get("REW_SCALE", 1)*reward,
                    done=new_done,
                    next_obs=new_obs,
                    q_val=q_vals,
                )
                return (new_obs, new_env_state, rng), (transition, info)

            rng, _rng = jax.random.split(rng)
            (*expl_state, rng), (transitions, infos) = jax.lax.scan(
                _step_env,
                (*expl_state, _rng),
                None,
                config["NUM_STEPS"],
            )
            expl_state = tuple(expl_state)

            train_state = train_state.replace(
                timesteps=train_state.timesteps
                + config["NUM_STEPS"] * config["NUM_ENVS"]
            )

            # BOOTSTRAP PHASE
            last_q = network.apply(
                {
                    "params": train_state.params,
                    "batch_stats": train_state.batch_stats,
                },
                transitions.next_obs[-1],
                train=False,
            )
            last_q = jnp.max(last_q, axis=-1)

            def _get_target(lambda_returns_and_next_q, transition):
                lambda_returns, next_q = lambda_returns_and_next_q
                target_bootstrap = (
                    transition.reward + config["GAMMA"] * (1 - transition.done) * next_q
                )
                delta = lambda_returns - next_q
                lambda_returns = (
                    target_bootstrap + config["GAMMA"] * config["LAMBDA"] * delta
                )
                lambda_returns = (
                    1 - transition.done
                ) * lambda_returns + transition.done * transition.reward
                next_q = jnp.max(transition.q_val, axis=-1)
                return (lambda_returns, next_q), lambda_returns

            last_q = last_q * (1 - transitions.done[-1])
            lambda_returns = transitions.reward[-1] + config["GAMMA"] * last_q
            _, targets = jax.lax.scan(
                _get_target,
                (lambda_returns, last_q),
                jax.tree_map(lambda x: x[:-1], transitions),
                reverse=True,
            )
            lambda_targets = jnp.concatenate((targets, lambda_returns[np.newaxis]))

            # Q-NETWORK UPDATE WITH REAL TRANSITIONS
            def _learn_epoch(carry, _):
                train_state, rng = carry

                def _learn_phase(carry, minibatch_and_target):

                    train_state, rng = carry
                    minibatch, target = minibatch_and_target

                    def _loss_fn(params):
                        q_vals, updates = network.apply(
                            {"params": params, "batch_stats": train_state.batch_stats},
                            minibatch.obs,
                            train=True,
                            mutable=["batch_stats"],
                        )

                        chosen_action_qvals = jnp.take_along_axis(
                            q_vals,
                            jnp.expand_dims(minibatch.action, axis=-1),
                            axis=-1,
                        ).squeeze(axis=-1)

                        loss = 0.5 * jnp.square(chosen_action_qvals - target).mean()
                        return loss, (updates, chosen_action_qvals)

                    (loss, (updates, qvals)), grads = jax.value_and_grad(
                        _loss_fn, has_aux=True
                    )(train_state.params)
                    train_state = train_state.apply_gradients(grads=grads)
                    train_state = train_state.replace(
                        grad_steps=train_state.grad_steps + 1,
                        batch_stats=updates["batch_stats"],
                    )
                    return (train_state, rng), (loss, qvals)

                def preprocess_transition(x, rng):
                    x = x.reshape(-1, *x.shape[2:])
                    x = jax.random.permutation(rng, x)
                    x = x.reshape(
                        config["NUM_MINIBATCHES"], -1, *x.shape[1:]
                    )
                    return x

                rng, _rng = jax.random.split(rng)
                minibatches = jax.tree_util.tree_map(
                    lambda x: preprocess_transition(x, _rng), transitions
                )
                targets = jax.tree_map(
                    lambda x: preprocess_transition(x, _rng), lambda_targets
                )

                rng, _rng = jax.random.split(rng)
                (train_state, rng), (loss, qvals) = jax.lax.scan(
                    _learn_phase, (train_state, rng), (minibatches, targets)
                )

                return (train_state, rng), (loss, qvals)

            rng, _rng = jax.random.split(rng)
            (train_state, rng), (loss, qvals) = jax.lax.scan(
                _learn_epoch, (train_state, rng), None, config["NUM_EPOCHS"]
            )

            train_state = train_state.replace(n_updates=train_state.n_updates + 1)
            metrics = {
                "env_step": train_state.timesteps,
                "update_steps": train_state.n_updates,
                "env_frame": train_state.timesteps * env.observation_space(env_params).shape[-1],
                "grad_steps": train_state.grad_steps,
                "td_loss": loss.mean(),
                "qvals": qvals.mean(),
            }
            metrics.update({k: v.mean() for k, v in infos.items()})

            if config.get("TEST_DURING_TRAINING", False):
                rng, _rng = jax.random.split(rng)
                test_metrics = jax.lax.cond(
                    train_state.n_updates
                    % int(config["NUM_UPDATES"] * config["TEST_INTERVAL"])
                    == 0,
                    lambda _: get_test_metrics(train_state, _rng),
                    lambda _: test_metrics,
                    operand=None,
                )
                if test_metrics is not None:
                    metrics.update({f"test_{k}": v for k, v in test_metrics.items()})

            # Update the world model
            def model_loss(params, batch_stats, obs, action, next_obs, reward):
                (next_obs_pred, reward_pred), updates = world_model.apply(
                    {"params": params, "batch_stats": batch_stats},
                    obs, action, train=True, mutable=["batch_stats"]
                )
                loss = ((next_obs_pred - next_obs)**2).mean() + ((reward_pred - reward)**2).mean()
                return loss, updates

            def preprocess_for_model(x):
                return x.reshape(-1, *x.shape[2:])

            obs_batch = preprocess_for_model(transitions.obs)
            action_batch = preprocess_for_model(transitions.action)
            next_obs_batch = preprocess_for_model(transitions.next_obs)
            reward_batch = preprocess_for_model(transitions.reward)

            (model_loss_val, updates), grads = jax.value_and_grad(model_loss, has_aux=True)(
                model_state.params, model_state.batch_stats,
                obs_batch, action_batch, next_obs_batch, reward_batch
            )

            model_state = model_state.apply_gradients(grads=grads)
            model_state = model_state.replace(batch_stats=updates["batch_stats"])
            metrics.update({"model_loss": model_loss_val})

            # --- PLANNING PHASE ---
            n = config.get("PLANNING_STEPS", 0)
            if n > 0:
                # According to the pseudocode:
                # s_i <- s'_i from the last environment step
                # s_0 <- s_i
                s_0 = transitions.next_obs[-1]  # This is s_i <- s'_i and thus s_0 <- s_i

                def planning_step(carry, _):
                    s_j, rng_planning, train_state = carry
                    rng_planning, rng_a = jax.random.split(rng_planning, 2)

                    # Qθ(s_j, ·)
                    q_vals_j = network.apply(
                        {
                            "params": train_state.params,
                            "batch_stats": train_state.batch_stats,
                        },
                        s_j,
                        train=False,
                    )

                    # a_j ~ Unif(A) with prob ε; otherwise argmax_a' Qθ(s_j,a')
                    eps = jnp.full(s_j.shape[0], eps_scheduler(train_state.n_updates))
                    a_j = jax.vmap(eps_greedy_exploration)(
                        jax.random.split(rng_a, s_j.shape[0]),
                        q_vals_j,
                        eps
                    )

                    # s'_j, r_j ~ P^S_φ(s_j,a_j), P^R_φ(s_j,a_j)
                    (s_prime_j, r_j), _ = world_model.apply(
                        {"params": model_state.params, "batch_stats": model_state.batch_stats},
                        s_j, a_j, train=False, mutable=[]
                    )

                    # In model planning, we have no terminal condition from the model.
                    # According to the pseudocode, we use 1(notterminal).
                    # Here, we assume no termination: done_j = 0
                    done_j = jnp.zeros_like(r_j)

                    # target_j <- r_j + γ * 1(notterminal) * max_{a'}Qθ(s'_j,a')
                    q_vals_next = network.apply(
                        {
                            "params": train_state.params,
                            "batch_stats": train_state.batch_stats,
                        },
                        s_prime_j,
                        train=False,
                    )
                    max_next_q = jnp.max(q_vals_next, axis=-1)
                    target_j = r_j + config["GAMMA"] * (1 - done_j) * max_next_q
                    target_j = jax.lax.stop_gradient(target_j)

                    def loss_fn(params):
                        q_vals_planning, updates = network.apply(
                            {"params": params, "batch_stats": train_state.batch_stats},
                            s_j,
                            train=True,
                            mutable=["batch_stats"],
                        )
                        chosen_q = jnp.take_along_axis(
                            q_vals_planning,
                            jnp.expand_dims(a_j, axis=-1),
                            axis=-1
                        ).squeeze(-1)
                        loss = 0.5 * jnp.square(chosen_q - target_j).mean()
                        return loss, (updates, chosen_q)

                    (loss_plan, (updates_plan, chosen_q_vals)), grads_plan = jax.value_and_grad(loss_fn, has_aux=True)(train_state.params)
                    train_state = train_state.apply_gradients(grads=grads_plan)
                    train_state = train_state.replace(
                        grad_steps=train_state.grad_steps + 1,
                        batch_stats=updates_plan["batch_stats"],
                    )

                    # s_j <- s'_j
                    s_j = s_prime_j

                    return (s_j, rng_planning, train_state), (loss_plan, chosen_q_vals)

                rng, rng_planning = jax.random.split(rng)
                (final_s_j, rng_planning, train_state), (planning_loss, planning_qvals) = jax.lax.scan(
                    planning_step,
                    (s_0, rng_planning, train_state),
                    None,
                    length=n
                )

                metrics["planning_loss"] = planning_loss.mean()
                metrics["planning_qvals"] = planning_qvals.mean()

            if config["WANDB_MODE"] != "disabled":
                def callback(metrics, original_rng):
                    wandb.log(metrics, step=metrics["update_steps"])

                jax.debug.callback(callback, metrics, original_rng)

            runner_state = (train_state, tuple(expl_state), test_metrics, model_state, rng)
            return runner_state, metrics

        rng, _rng = jax.random.split(rng)
        test_metrics = get_test_metrics(train_state, _rng)

        rng, _rng = jax.random.split(rng)
        expl_state = vmap_reset(config["NUM_ENVS"])(_rng)

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, expl_state, test_metrics, model_state, _rng)

        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )

        return {"runner_state": runner_state, "metrics": metrics}

    return train



def single_run(config):

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
    train_vjit = jax.jit(jax.vmap(make_train(config)))
    outs = jax.block_until_ready(train_vjit(rngs))
    print(f"Took {time.time()-t0} seconds to complete.")

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

        for i, rng in enumerate(rngs):
            params = jax.tree_map(lambda x: x[i], model_state.params)
            save_path = os.path.join(
                save_dir,
                f'{alg_name}_{env_name}_seed{config["SEED"]}_vmap{i}.safetensors',
            )
            save_params(params, save_path)


def tune(default_config):
    """Hyperparameter sweep with wandb."""

    default_config = {**default_config, **default_config["alg"]}
    print(default_config)
    alg_name = default_config.get("ALG_NAME", "pqn")
    env_name = default_config["ENV_NAME"]

    def wrapped_make_train():
        wandb.init(project=default_config["PROJECT"])

        config = copy.deepcopy(default_config)
        for k, v in dict(wandb.config).items():
            config[k] = v

        print("running experiment with params:", config)

        rng = jax.random.PRNGKey(config["SEED"])
        rngs = jax.random.split(rng, config["NUM_SEEDS"])
        train_vjit = jax.jit(jax.vmap(make_train(config)))
        outs = jax.block_until_ready(train_vjit(rngs))

    sweep_config = {
        "name": f"{alg_name}_{env_name}",
        "method": "bayes",
        "metric": {
            "name": "test_returned_episode_returns",
            "goal": "maximize",
        },
        "parameters": {
            "LR": {
                "values": [
                    0.001,
                    0.0005,
                    0.0001,
                    0.00005,
                ]
            },
        },
    }

    wandb.login()
    sweep_id = wandb.sweep(
        sweep_config, entity=default_config["ENTITY"], project=default_config["PROJECT"]
    )
    wandb.agent(sweep_id, wrapped_make_train, count=1000)


@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(config):
    config = OmegaConf.to_container(config)
    print("Config:\n", OmegaConf.to_yaml(config))
    if config["HYP_TUNE"]:
        tune(config)
    else:
        single_run(config)


if __name__ == "__main__":
    main()

