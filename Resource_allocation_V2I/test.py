import dataclasses
import os
from typing import List

import numpy as np


from Resource_allocation_V2I import Environment3, dataStruct
from Resource_allocation_V2I.main_train import down_lanes, up_lanes, left_lanes, width, right_lanes, height, n_veh, \
    BS_width, n_interference_vehicle
"""Time slot related."""
time_slot_start: int = 0
time_slot_end: int = 299
time_slot_number: int = 300
time_slot_length: int = 1
if __name__ == "__main__":
    # env = Environment3.Environ(down_lanes, up_lanes, left_lanes, right_lanes, width, height, n_veh, n_interference_vehicle, BS_width)
    # env.init_time_slots()
    # env.generate_vehicles_by_number()
    # v=env.vehicles
    # for i in range(3):
    #     t = v[0].get_vehicle_location(i)
    #     print(t)
    #
    # print("test")

    # time_slots: dataStruct.timeSlots = dataStruct.timeSlots(
    #     start=time_slot_start,
    #     end=time_slot_end,
    #     slot_length=time_slot_length,
    # )
    # print(time_slots)


    env = Environment3.Environ(down_lanes, up_lanes, left_lanes, right_lanes, width, height, n_veh, n_interference_vehicle, BS_width)
    env.new_random_game()
    print("test")