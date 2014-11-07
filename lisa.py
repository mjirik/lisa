#! /opt/local/bin/python
# -*- coding: utf-8 -*-

""" Run Lisa. """

# import funkcí z jiného adresáře
# import sys
# import os.path

# path_to_script = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.join(path_to_script, "./src"))

if __name__ == "__main__":

    import lisa.organ_segmentation as osegg
    osegg.main()
    exit()
