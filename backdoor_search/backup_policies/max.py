from backup_policies.base import BackupPolicy


class MaxBackupPolicy(BackupPolicy):
    def propagate(self, node, reward):
        node.Q = max([node.Q, reward])