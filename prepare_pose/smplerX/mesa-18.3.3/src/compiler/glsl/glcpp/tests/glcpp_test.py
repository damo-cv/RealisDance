# encoding=utf-8
# Copyright © 2018 Intel Corporation

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Run glcpp tests with various line endings."""

from __future__ import print_function
import argparse
import difflib
import io
import os
import subprocess
import sys
import tempfile


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('glcpp', help='Path to the he glcpp binary.')
    parser.add_argument('testdir', help='Path to tests and expected output.')
    parser.add_argument('--unix', action='store_true', help='Run tests for Unix style newlines')
    parser.add_argument('--windows', action='store_true', help='Run tests for Windows/Dos style newlines')
    parser.add_argument('--oldmac', action='store_true', help='Run tests for Old Mac (pre-OSX) style newlines')
    parser.add_argument('--bizarro', action='store_true', help='Run tests for Bizarro world style newlines')
    parser.add_argument('--valgrind', action='store_true', help='Run with valgrind for errors')
    return parser.parse_args()


def parse_test_file(filename, nl_format):
    """Check for any special arguments and return them as a list."""
    # Disable "universal newlines" mode; we can't directly use `nl_format` as
    # the `newline` argument, because the "bizarro" test uses something Python
    # considers invalid.
    with io.open(filename, newline='') as f:
        for l in f.read().split(nl_format):
            if 'glcpp-args:' in l:
                return l.split('glcpp-args:')[1].strip().split()
    return []


def test_output(glcpp, filename, expfile, nl_format='\n'):
    """Test that the output of glcpp is what we expect."""
    extra_args = parse_test_file(filename, nl_format)

    with open(filename, 'rb') as f:
        proc = subprocess.Popen(
            [glcpp] + extra_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.PIPE)
        actual, _ = proc.communicate(f.read())
        actual = actual.decode('utf-8')

    with open(expfile, 'r') as f:
        expected = f.read()

    if actual == expected:
        return (True, [])
    return (False, difflib.unified_diff(actual.splitlines(), expected.splitlines()))


def _valgrind(glcpp, filename):
    """Run valgrind and report any warnings."""
    extra_args = parse_test_file(filename, nl_format='\n')

    try:
        fd, tmpfile = tempfile.mkstemp()
        os.close(fd)
        with open(filename, 'rb') as f:
            proc = subprocess.Popen(
                ['valgrind', '--error-exitcode=31', '--log-file', tmpfile, glcpp] + extra_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.PIPE)
            proc.communicate(f.read())
            if proc.returncode != 31:
                return (True, [])
        with open(tmpfile, 'rb') as f:
            contents = f.read()
        return (False, contents)
    finally:
        os.unlink(tmpfile)


def test_unix(args):
    """Test files with unix style (\n) new lines."""
    total = 0
    passed = 0

    print('============= Testing for Correctness (Unix) =============')
    for filename in os.listdir(args.testdir):
        if not filename.endswith('.c'):
            continue

        print(   '{}:'.format(os.path.splitext(filename)[0]), end=' ')
        total += 1

        testfile = os.path.join(args.testdir, filename)
        valid, diff = test_output(args.glcpp, testfile, testfile + '.expected')
        if valid:
            passed += 1
            print('PASS')
        else:
            print('FAIL')
            for l in diff:
                print(l, file=sys.stderr)

    if not total:
        raise Exception('Could not find any tests.')

    print('{}/{}'.format(passed, total), 'tests returned correct results')
    return total == passed


def _replace_test(args, replace):
    """Test files with non-unix style line endings. Print your own header."""
    total = 0
    passed = 0

    for filename in os.listdir(args.testdir):
        if not filename.endswith('.c'):
            continue

        print(   '{}:'.format(os.path.splitext(filename)[0]), end=' ')
        total += 1
        testfile = os.path.join(args.testdir, filename)
        try:
            fd, tmpfile = tempfile.mkstemp()
            os.close(fd)
            with io.open(testfile, 'rt') as f:
                contents = f.read()
            with io.open(tmpfile, 'wt') as f:
                f.write(contents.replace('\n', replace))
            valid, diff = test_output(
                args.glcpp, tmpfile, testfile + '.expected', nl_format=replace)
        finally:
            os.unlink(tmpfile)

        if valid:
            passed += 1
            print('PASS')
        else:
            print('FAIL')
            for l in diff:
                print(l, file=sys.stderr)

    if not total:
        raise Exception('Could not find any tests.')

    print('{}/{}'.format(passed, total), 'tests returned correct results')
    return total == passed


def test_windows(args):
    """Test files with windows/dos style (\r\n) new lines."""
    print('============= Testing for Correctness (Windows) =============')
    return _replace_test(args, '\r\n')


def test_oldmac(args):
    """Test files with Old Mac style (\r) new lines."""
    print('============= Testing for Correctness (Old Mac) =============')
    return _replace_test(args, '\r')


def test_bizarro(args):
    """Test files with Bizarro world style (\n\r) new lines."""
    # This is allowed by the spec, but why?
    print('============= Testing for Correctness (Bizarro) =============')
    return _replace_test(args, '\n\r')


def test_valgrind(args):
    total = 0
    passed = 0

    print('============= Testing for Valgrind Warnings =============')
    for filename in os.listdir(args.testdir):
        if not filename.endswith('.c'):
            continue

        print(   '{}:'.format(os.path.splitext(filename)[0]), end=' ')
        total += 1
        valid, log = _valgrind(args.glcpp, os.path.join(args.testdir, filename))
        if valid:
            passed += 1
            print('PASS')
        else:
            print('FAIL')
            print(log, file=sys.stderr)

    if not total:
        raise Exception('Could not find any tests.')

    print('{}/{}'.format(passed, total), 'tests returned correct results')
    return total == passed


def main():
    args = arg_parser()

    success = True
    if args.unix:
        success = success and test_unix(args)
    if args.windows:
        success = success and test_windows(args)
    if args.oldmac:
        success = success and test_oldmac(args)
    if args.bizarro:
        success = success and test_bizarro(args)
    if args.valgrind:
        success = success and test_valgrind(args)

    exit(0 if success else 1)


if __name__ == '__main__':
    main()
