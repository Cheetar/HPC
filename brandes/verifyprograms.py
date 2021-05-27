import logging
import os
import subprocess
import signal
import shutil
from collections import OrderedDict

from fulp_compare import fulpdiff


class Alarm(Exception):
    pass


def alarm_handler(signum, frame):
    raise Alarm


def gen_header(files):
    all_head = "login,tot_err"
    for file in files:
        all_head += f",{file}"
    return all_head


def read_brandes_scores(filename):
    scores = OrderedDict()
    with open(filename) as f:
        for node, score in enumerate(f):
            scores[node] = float(score)
    return scores


def compare_brandes_scores(ref, act, max_fulp_difference=100):
    for ref_key, ref_value in ref.items():
        if ref_key not in act.keys():
            return (False, f"key {ref_key} not found")
        act_val = act[ref_key]
        fulp_diff = fulpdiff(ref_value, act_val)
        if ((ref_value > 10e-5 and fulp_diff > max_fulp_difference)
            or (ref_value < 10e-4 and act[ref_key] > 10e-4)):
            return (False, f"key {ref_key} expected: {ref_value} found: {act_val}")
    return (True, "")


log = logging.getLogger("verifyprograms")


def get_run_number(filebase="results"):
    run_number = 1
    while os.path.isfile(f"{filebase}-run{run_number}.csv"):
        run_number += 1
    return run_number


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    run_number = get_run_number()
    fh = logging.FileHandler(f'verifyprograms-run{run_number}.log')
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fh.setFormatter(formatter)
    log.addHandler(fh)
    sink = open('/dev/null', 'w')
    tests = [f"gen{x}" for x in range(1, 10)]
    timelimit = 10
    # --------------------------------------------------
    # TODO: put the directory with your source code here
    dirs = ['entropy']
    # --------------------------------------------------
    basedir = os.getcwd()
    with open(f'results-run{run_number}.csv', 'a') as results_stat:
        results_stat.write(gen_header(tests))
        for subdir in dirs:
            results_stat.flush()
            error_count = 0
            os.chdir(basedir)
            os.chdir(subdir)
            log.info("dir: %s" % subdir)
            results_stat.write("\n"+subdir+",")
            shutil.rmtree("build", ignore_errors=True)
            retcode = subprocess.call(("make", "clean"), stdout=sink, stderr=sink)
            if retcode != 0:
                log.error("make clean doesn't work")
                continue
            retcode = subprocess.call("make", stdout=sink, stderr=sink)
            if retcode != 0:
                log.error("make doesn't work")
                continue
            test_counter = 0
            res_line = ""
            with open('errors.txt', 'w') as error_file:
                for test_file in tests:
                    params = ["./brandes", f"../tests/{test_file}.txt", f"res-{test_file}.txt"]
                    compact_params = " ".join(params)
                    log.info(f"dir: {subdir} params: {compact_params}")
                    test_counter += 1

                    signal.signal(signal.SIGALRM, alarm_handler)
                    signal.alarm(timelimit)
                    child = None
                    try:
                        #child = subprocess.Popen(params, stdout=sink, stderr=sink)
                        #outcode = child.wait()
                        outcode = 0
                        signal.alarm(0)
                        log.info("subprocess finished")
                        if outcode != 0:
                            log.error("non-0 exit code")
                            res_line +="non0, "
                            error_count +=1
                            error_file.write(f"{compact_params}\n")
                            continue
                    except FileNotFoundError:
                        log.error("exec not found")
                        res_line +="notfound, "
                        error_count += 1
                        error_file.write(f"{compact_params}\n")
                        continue
                    except Alarm:
                        log.info("timeout!")
                        if child is not None:
                            log.info("killing the child")
                            child.kill()
                            log.info("child killed")
                        error_count += 1
                        res_line += "time, "
                        error_file.write(f"{compact_params}\n")
                        continue

                    ref_result = read_brandes_scores(f"../tests/res-{test_file}.txt")
                    prog_result = None
                    try:
                        prog_result = read_brandes_scores(f"res-{test_file}.txt")
                    except (ValueError, IOError, IndexError):
                        log.info(f"error while reading output: {e}")
                        res_line += "out, "
                        error_count += 1
                        error_file.write(f"{compact_params}\n")
                        continue
                    (comp, error) = compare_brandes_scores(ref_result, prog_result)
                    if not comp:
                        log.info(f"error in the result: {error}"),
                        res_line += "res, "
                        error_file.write(f"{compact_params}\n")
                        error_count += 1
                    else:
                        res_line += "OK, "

            log.info("subdir %s errors %d" % (subdir, error_count))
            results_stat.write(f"{error_count},{res_line}")
        results_stat.write("\n")

