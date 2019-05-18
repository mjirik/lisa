import json
import appdirs
import pathlib
import urllib
import urllib.request
import os.path as op
import sys
import os

# print(op.dirname(os.path.abspath(__file__)))
# print(op.join(op.dirname(os.path.abspath(__file__)), "../../bodynavigation/"))
# sys.path.append(op.join(op.dirname(os.path.abspath(__file__)), "../../bodynavigation/"))
#
# import bodynavigation

appname = "LisaMreImport"
appauthor = "mjirik"
dirs = appdirs.AppDirs(appname=appname, appauthor=appauthor)
fn_config_dir = pathlib.Path(dirs.user_config_dir)
fn_config = fn_config_dir / "config.json"
fn_config_dir.mkdir(exist_ok=True, parents=True)
fn_data_dir = pathlib.Path(dirs.user_data_dir)
fn_data_dir.mkdir(exist_ok=True, parents=True)
fn_data = pathlib.Path(dirs.user_config_dir) / "data.json"

# dir for output files
lisa_data = pathlib.Path("~/lisa_data").expanduser()
lisa_data.mkdir(exist_ok=True, parents=True)

with open(fn_config, 'r') as infile:
    data_config = json.load(infile)

url = data_config["url"]
url_list = url + "list.php"

x = urllib.request.urlopen(url)
text = x.read().decode()
# text_lines = text.split("\n")
text_parsered = [text_line.split(" ") for text_line in text.split("\n")]

filenames = [line[0] for line in text_parsered if len(line) > 2]
dates = [line[1] for line in text_parsered if len(line) > 2]
times = [line[2] for line in text_parsered if len(line) > 2]

# table_data = list(zip(*text_parsered))
table_dict = dict(zip(["filename", "date", "time"], [filenames, dates, times]))

# print(list(table_dict))

# data_out = filenames


# with first run just make the list of actual files
if fn_data.exists():
    with open(fn_data, 'r') as infile:
        filenames_local = json.load(infile)
    # set difference
    autolisa_paths = []
    for filename in list(set(filenames) - set(filenames_local)):
        url_file = url + filename
        fn_local = lisa_data / filename
        if not fn_local.exists():
            print("Downloading from {} to {}".format(url_file, fn_local))
            fn_local2, headers = urllib.request.urlretrieve(url_file, filename=fn_local)
        #fn_local and fn_local2 are probably the same
        autolisa_paths.append(fn_local)

    print("Running Auto-Lisa")
    try:
        from lisa import autolisa
        al = autolisa.AutoLisa()
        al.run_in_paths(autolisa_paths)
    except Exception as e:
        import traceback
        traceback.print_exc()

print(fn_data)
# with open(fn_data, 'w') as outfile:
#     json.dump(filenames, outfile)

