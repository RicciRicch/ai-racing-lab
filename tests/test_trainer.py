import csv

import numpy as np
import pytest



def framework_param(framework, module):
    try:
        __import__(module)
        marks = ()
    except ImportError:
        marks = pytest.mark.skip(reason=f"{module} not installed")
    return pytest.param(framework, marks=marks, id=framework)


FRAMEWORKS = [
    framework_param("torch", "torch"),
    framework_param("tf", "tensorflow"),
]

EXTENSIONS = {"torch": ".pt", "tf": ".keras"}

from rl.common import PPOConfig
from rl.evaluate import load_policy
from rl.trainer import main, make_agent, parse_args, train


def tiny_config(**overrides):
    values = dict(
        tracks="train",
        num_envs=2,
        num_steps=32,
        total_timesteps=256,
        num_minibatches=2,
        update_epochs=2,
        max_steps=50,
        eval_interval=2,
        eval_episodes=1,
    )
    values.update(overrides)
    return PPOConfig(**values)


def read_csv(path):
    with open(path, newline="") as file:
        return list(csv.DictReader(file))


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_training_run_writes_logs_and_checkpoints(tmp_path, framework):
    config = tiny_config()
    agent = make_agent(framework, config)

    best = train(agent, config, str(tmp_path), log=lambda message: None)

    assert np.isfinite(best)

    progress = read_csv(tmp_path / "progress.csv")
    assert len(progress) == config.num_updates == 4
    assert int(progress[-1]["global_step"]) == 256

    evaluations = read_csv(tmp_path / "eval.csv")
    # Updates 2 and 4, three training tracks each.
    assert len(evaluations) == 6

    assert PPOConfig.load(tmp_path / "config.json") == config

    for name in ("best", "final"):
        policy = load_policy(str(tmp_path / (name + EXTENSIONS[framework])))
        assert policy(np.zeros(6, np.float32)).shape == (2,)


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_learning_rate_is_annealed(tmp_path, framework):
    config = tiny_config(eval_interval=100)
    train(make_agent(framework, config), config, str(tmp_path), log=lambda m: None)

    rates = [float(r["learning_rate"]) for r in read_csv(tmp_path / "progress.csv")]

    assert rates[0] == pytest.approx(config.learning_rate)
    assert rates == sorted(rates, reverse=True)


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_same_seed_gives_same_training(tmp_path, framework):
    results = []

    for run in ("a", "b"):
        config = tiny_config(eval_interval=100)
        train(make_agent(framework, config), config, str(tmp_path / run),
              log=lambda m: None)
        results.append(read_csv(tmp_path / run / "progress.csv"))

    keys = ("episode_return", "policy_loss", "value_loss")
    assert [[r[k] for k in keys] for r in results[0]] == \
        [[r[k] for k in keys] for r in results[1]]


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_cli_trains_with_each_framework(tmp_path, framework):
    main([
        "--framework", framework,
        "--runs-dir", str(tmp_path), "--run-name", "cli",
        "--num-envs", "2", "--num-steps", "16", "--total-timesteps", "64",
        "--max-steps", "20", "--eval-episodes", "1",
    ])

    assert (tmp_path / "cli" / ("final" + EXTENSIONS[framework])).exists()


def test_cli_maps_options_to_config():
    args = parse_args([
        "--tracks", "train:1", "--total-timesteps", "128",
        "--anneal-lr", "false", "--learning-rate", "0.001",
    ])

    assert args.tracks == "train:1"
    assert args.total_timesteps == 128
    assert args.anneal_lr is False
    assert args.learning_rate == 0.001

