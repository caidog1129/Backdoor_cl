from reward_functions.base import RewardFunction


class TreeWeightReward(RewardFunction):
    def __call__(self, instance, **kwargs):
        return kwargs['tree_weight']