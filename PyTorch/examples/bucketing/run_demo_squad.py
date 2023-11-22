# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company

from datasets_library import squad
from plotting import plotter


if __name__ == '__main__':
    print("Plotting squad, takes 2-3 mins to run")
    plotter([squad(1), squad(4), squad(16), squad(64), squad(256), squad(512)], 'squad.svg', ['bs='+str(bs) for bs in [1,4,16,64,256,512]])