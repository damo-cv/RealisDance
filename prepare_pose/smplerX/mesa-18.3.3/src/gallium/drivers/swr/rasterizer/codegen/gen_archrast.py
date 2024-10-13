# Copyright (C) 2014-2016 Intel Corporation.   All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice (including the next
# paragraph) shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.

# Python source
from __future__ import print_function
import os
import sys
import re
from gen_common import *

def parse_event_fields(lines, idx, event_dict):
    field_names = []
    field_types = []
    end_of_event = False

    num_fields = 0

    # record all fields in event definition.
    # note: we don't check if there's a leading brace.
    while not end_of_event and idx < len(lines):
        line = lines[idx].rstrip()
        idx += 1

        field = re.match(r'(\s*)(\w+)(\s*)(\w+)', line)

        if field:
            field_types.append(field.group(2))
            field_names.append(field.group(4))
            num_fields += 1

        end_of_event = re.match(r'(\s*)};', line)

    event_dict['field_types'] = field_types
    event_dict['field_names'] = field_names
    event_dict['num_fields'] = num_fields

    return idx

def parse_enums(lines, idx, event_dict):
    enum_names = []
    end_of_enum = False

    # record all enum values in enumeration
    # note: we don't check if there's a leading brace.
    while not end_of_enum and idx < len(lines):
        line = lines[idx].rstrip()
        idx += 1

        preprocessor = re.search(r'#if|#endif', line)

        if not preprocessor:
            enum = re.match(r'(\s*)(\w+)(\s*)', line)

            if enum:
                enum_names.append(line)

            end_of_enum = re.match(r'(\s*)};', line)

    event_dict['names'] = enum_names
    return idx

def parse_protos(protos, filename):

    with open(filename, 'r') as f:
        lines=f.readlines()

        idx = 0

        eventId = 0
        raw_text = []
        while idx < len(lines):
            line = lines[idx].rstrip()
            idx += 1

            # search for event definitions.
            match = re.match(r'(\s*)event(\s*)(\w+)', line)

            if match:
                eventId += 1
                event_name = match.group(3)
                protos['event_names'].append(event_name)

                protos['events'][event_name] = {}
                protos['events'][event_name]['event_id'] = eventId
                idx = parse_event_fields(lines, idx, protos['events'][event_name])

            # search for enums.
            match = re.match(r'(\s*)enum(\s*)(\w+)', line)

            if match:
                enum_name = match.group(3)
                protos['enum_names'].append(enum_name)

                protos['enums'][enum_name] = {}
                idx = parse_enums(lines, idx, protos['enums'][enum_name])

def main():

    # Parse args...
    parser = ArgumentParser()
    parser.add_argument('--proto', '-p', help='Path to proto file', required=True)
    parser.add_argument('--proto_private', '-pp', help='Path to private proto file', required=True)
    parser.add_argument('--output', '-o', help='Output filename (i.e. event.hpp)', required=True)
    parser.add_argument('--gen_event_hpp', help='Generate event header', action='store_true', default=False)
    parser.add_argument('--gen_event_cpp', help='Generate event cpp', action='store_true', default=False)
    parser.add_argument('--gen_eventhandler_hpp', help='Generate eventhandler header', action='store_true', default=False)
    parser.add_argument('--gen_eventhandlerfile_hpp', help='Generate eventhandler header for writing to files', action='store_true', default=False)
    args = parser.parse_args()

    proto_filename = args.proto
    proto_private_filename = args.proto_private

    (output_dir, output_filename) = os.path.split(args.output)

    if not output_dir:
        output_dir = '.'

    #print('output_dir = %s' % output_dir, file=sys.stderr)
    #print('output_filename = %s' % output_filename, file=sys.stderr)

    if not os.path.exists(proto_filename):
        print('Error: Could not find proto file %s' % proto_filename, file=sys.stderr)
        return 1

    if not os.path.exists(proto_private_filename):
        print('Error: Could not find private proto file %s' % proto_private_filename, file=sys.stderr)
        return 1

    final_output_dir = output_dir
    MakeDir(final_output_dir)
    output_dir = MakeTmpDir('_codegen')

    protos = {}
    protos['events'] = {}       # event dictionary containing events with their fields
    protos['event_names'] = []  # needed to keep events in order parsed. dict is not ordered.
    protos['enums'] = {}
    protos['enum_names'] = []

    parse_protos(protos, proto_filename)
    parse_protos(protos, proto_private_filename)

    rval = 0

    try:
        # Generate event header
        if args.gen_event_hpp:
            curdir = os.path.dirname(os.path.abspath(__file__))
            template_file = os.sep.join([curdir, 'templates', 'gen_ar_event.hpp'])
            output_fullpath = os.sep.join([output_dir, output_filename])

            MakoTemplateWriter.to_file(template_file, output_fullpath,
                    cmdline=sys.argv,
                    filename=output_filename,
                    protos=protos)

        # Generate event implementation
        if args.gen_event_cpp:
            curdir = os.path.dirname(os.path.abspath(__file__))
            template_file = os.sep.join([curdir, 'templates', 'gen_ar_event.cpp'])
            output_fullpath = os.sep.join([output_dir, output_filename])

            MakoTemplateWriter.to_file(template_file, output_fullpath,
                    cmdline=sys.argv,
                    filename=output_filename,
                    protos=protos)

        # Generate event handler header
        if args.gen_eventhandler_hpp:
            curdir = os.path.dirname(os.path.abspath(__file__))
            template_file = os.sep.join([curdir, 'templates', 'gen_ar_eventhandler.hpp'])
            output_fullpath = os.sep.join([output_dir, output_filename])

            MakoTemplateWriter.to_file(template_file, output_fullpath,
                    cmdline=sys.argv,
                    filename=output_filename,
                    event_header='gen_ar_event.hpp',
                    protos=protos)

        # Generate event handler header
        if args.gen_eventhandlerfile_hpp:
            curdir = os.path.dirname(os.path.abspath(__file__))
            template_file = os.sep.join([curdir, 'templates', 'gen_ar_eventhandlerfile.hpp'])
            output_fullpath = os.sep.join([output_dir, output_filename])

            MakoTemplateWriter.to_file(template_file, output_fullpath,
                    cmdline=sys.argv,
                    filename=output_filename,
                    event_header='gen_ar_eventhandler.hpp',
                    protos=protos)

        rval = CopyDirFilesIfDifferent(output_dir, final_output_dir)

    except:
        rval = 1

    finally:
        DeleteDirTree(output_dir)

    return rval

if __name__ == '__main__':
    sys.exit(main())

