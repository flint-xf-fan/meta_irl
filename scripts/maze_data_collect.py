import argparse

import tensorflow.compat.v1 as tf

from inverse_rl.algos.trpo import TRPO
from inverse_rl.models.tf_util import get_session_config
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline

from inverse_rl.envs.env_utils import CustomGymEnv
from inverse_rl.utils.log_utils import rllab_logdir
from inverse_rl.utils.hyper_sweep import run_sweep_parallel, run_sweep_serial

parser = argparse.ArgumentParser()
parser.add_argument('--pre_epoch', type=int, default=1000)
parser.add_argument('--n_itr', type=int, default=3000)
parser.add_argument('--turn_on_wandb', action='store_true')
parser.add_argument('--wandb_entity', type=str, default='sff1019')
parser.add_argument('--wandb_project',
                    type=str,
                    default='reinforcement_learning_algorithms')
parser.add_argument('--wandb_run_name',
                    type=str,
                    default='metairl-maze_wall_collect_trpo')
parser.add_argument('--wandb_monitor_gym', action='store_true')
args = parser.parse_args()


def main(exp_name, ent_wt=1.0, discrete=True):
    tf.reset_default_graph()
    if discrete:
        env = TfEnv(
            CustomGymEnv('PointMazeLeft-v0',
                         record_video=False,
                         record_log=False))
    else:
        env = TfEnv(
            CustomGymEnv('PointMazeLeftCont-v0',
                         record_video=False,
                         record_log=False))

    policy = GaussianMLPPolicy(name='policy',
                               env_spec=env.spec,
                               hidden_sizes=(32, 32))
    with tf.Session(config=get_session_config()) as sess:
        algo = TRPO(
            env=env,
            sess=sess,
            policy=policy,
            n_itr=2000,
            batch_size=20000,
            max_path_length=500,
            discount=0.99,
            store_paths=True,
            entropy_weight=ent_wt,
            baseline=LinearFeatureBaseline(env_spec=env.spec),
            exp_name=exp_name,
            turn_on_wandb=args.turn_on_wandb,
            render_env=True,
            gif_dir='logs/maze_wall_meta_irl',
            gif_header='',
            wandb_entity=args.wandb_entity,
            wandb_project=args.wandb_project,
            wandb_run_name=args.wandb_run_name,
            wandb_monitor_gym=args.wandb_monitor_gym,
        )
        if discrete:
            output = 'data/maze_left_data_collect_discrete-15/%s' % exp_name
        else:
            output = 'data/maze_left_data_collect/%s' % exp_name
    with rllab_logdir(algo=algo, dirname=output):
        algo.train()


if __name__ == "__main__":
    params_dict = {
        'ent_wt': [0.1],
        'discrete': [
            True
            # False
        ]  # Setting discrete to 'True' to get training data, 'False' to get test data (test unseen positions)
    }
    run_sweep_parallel(main, params_dict, repeat=1)
