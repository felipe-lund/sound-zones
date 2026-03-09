import numpy as np
import pyroomacoustics as pra

room_dim = [5, 5, 5]
room = pra.ShoeBox(room_dim, fs=16_000, absorption=0.2, max_order=15)

room.plot()
