import numpy as np

class StateProximator:
    def __init__(self, size):
        self.size = size
        self.state_buffer = np.zeros((size, 8))

    def add(self, observation):
        self.state_buffer[:-1] = self.state_buffer[1:]
        self.state_buffer[-1] = observation

def test_state_proximator():
    # Initialize StateProximator with size 3
    state_proximator = StateProximator(3)

    # Add observations to the state buffer
    state_proximator.add(np.array([1, 2, 3, 4, 5, 6, 7, 8]))
    state_proximator.add(np.array([2, 3, 4, 5, 6, 7, 8, 9]))
    state_proximator.add(np.array([3, 4, 5, 6, 7, 8, 9, 10]))

    # Check that the state buffer has the correct shape and values
    assert state_proximator.state_buffer.shape == (3, 8)
    assert np.allclose(state_proximator.state_buffer, np.array([[1, 2, 3, 4, 5, 6, 7, 8],
                                                               [2, 3, 4, 5, 6, 7, 8, 9],
                                                               [3, 4, 5, 6, 7, 8, 9, 10]]))

    # Add another observation to the state buffer
    state_proximator.add(np.array([4, 5, 6, 7, 8, 9, 10, 11]))

    # Check that the state buffer has been updated correctly
    assert np.allclose(state_proximator.state_buffer, np.array([[2, 3, 4, 5, 6, 7, 8, 9],
                                                               [3, 4, 5, 6, 7, 8, 9, 10],
                                                               [4, 5, 6, 7, 8, 9, 10, 11]]))