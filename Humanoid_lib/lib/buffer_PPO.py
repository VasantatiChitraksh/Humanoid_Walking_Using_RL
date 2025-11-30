import torch


class PPOBuffer:
    def __init__(self, observation_shape, action_shape, buffer_size, number_of_environments, computation_device, discount_factor=0.99, gae_smoothing_factor=0.95):
        self.buffer_size = buffer_size
        self.observation_buffer = torch.zeros(
            (buffer_size, number_of_environments, *observation_shape), dtype=torch.float32, device=computation_device)
        self.action_buffer = torch.zeros(
            (buffer_size, number_of_environments, *action_shape), dtype=torch.float32, device=computation_device)
        self.reward_buffer = torch.zeros(
            (buffer_size, number_of_environments), dtype=torch.float32, device=computation_device)
        self.value_buffer = torch.zeros(
            (buffer_size, number_of_environments), dtype=torch.float32, device=computation_device)
        self.terminated_buffer = torch.zeros(
            (buffer_size, number_of_environments), dtype=torch.float32, device=computation_device)
        self.truncated_buffer = torch.zeros(
            (buffer_size, number_of_environments), dtype=torch.float32, device=computation_device)
        self.log_probability_buffer = torch.zeros(
            (buffer_size, number_of_environments), dtype=torch.float32, device=computation_device)
        self.discount_factor, self.gae_smoothing_factor = discount_factor, gae_smoothing_factor
        self.buffer_pointer = 0

    def store(self, observation, action, reward, value_estimate, terminated_flag, truncated_flag, log_probability):
        self.observation_buffer[self.buffer_pointer] = observation
        self.action_buffer[self.buffer_pointer] = action
        self.reward_buffer[self.buffer_pointer] = reward
        self.value_buffer[self.buffer_pointer] = value_estimate
        self.terminated_buffer[self.buffer_pointer] = terminated_flag
        self.truncated_buffer[self.buffer_pointer] = truncated_flag
        self.log_probability_buffer[self.buffer_pointer] = log_probability
        self.buffer_pointer += 1

    def calculate_advantages(self, last_value_estimates, last_terminated_flags, last_truncated_flags):
        assert self.buffer_pointer == self.buffer_size

        with torch.no_grad():
            advantage_buffer = torch.zeros_like(self.reward_buffer)
            current_gae_value = 0.0
            for time_step in reversed(range(self.buffer_size)):
                if time_step == self.buffer_size - 1:
                    next_value_estimates = last_value_estimates
                    termination_mask = 1.0 - last_terminated_flags
                    truncation_mask = 1.0 - last_truncated_flags
                else:
                    next_value_estimates = self.value_buffer[time_step + 1]
                    termination_mask = 1.0 - \
                        self.terminated_buffer[time_step + 1]
                    truncation_mask = 1.0 - \
                        self.truncated_buffer[time_step + 1]

                temporal_difference_error = self.reward_buffer[time_step] + self.discount_factor * \
                    next_value_estimates * termination_mask - \
                    self.value_buffer[time_step]
                current_gae_value = temporal_difference_error + self.discount_factor * \
                    self.gae_smoothing_factor * termination_mask * \
                    truncation_mask * current_gae_value
                advantage_buffer[time_step] = current_gae_value

            return_buffer = advantage_buffer + self.value_buffer
            return advantage_buffer, return_buffer

    def get(self):
        assert self.buffer_pointer == self.buffer_size
        self.buffer_pointer = 0
        return self.observation_buffer, self.action_buffer, self.log_probability_buffer
