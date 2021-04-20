import os


class GetEnvs:
    def __init__(self, list_env):
        self.list_env = list_env
        self.env_dict = {}

    def load_envs(self, list_env=None):
        if list_env is not None:
            self.list_env = list_env
        self.env_dict = {}

        for env in self.list_env:
            temp_env_variable = os.environ.get(env)
            if temp_env_variable is None:
                return None

            else:
                self.env_dict[env] = temp_env_variable
        return True

    def get_env_vars(self, var_name=None):

        if len(self.env_dict) == 0:
            return None

        if var_name is not None:
            return self.env_dict[var_name]

        return self.env_dict
