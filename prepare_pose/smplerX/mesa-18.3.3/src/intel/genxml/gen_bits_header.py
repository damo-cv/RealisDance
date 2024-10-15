#encoding=utf-8
# Copyright © 2017 Intel Corporation

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

from __future__ import (
    absolute_import, division, print_function, unicode_literals
)

import argparse
import os
import xml.parsers.expat

from mako.template import Template

TEMPLATE = Template("""\
<%!
from operator import itemgetter
%>\
/*
 * Copyright © 2017 Intel Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

/* THIS FILE HAS BEEN GENERATED, DO NOT HAND EDIT.
 *
 * Sizes of bitfields in genxml instructions, structures, and registers.
 */

#ifndef ${guard}
#define ${guard}

#include <stdint.h>

#include "dev/gen_device_info.h"
#include "util/macros.h"

<%def name="emit_per_gen_prop_func(item, prop)">
%if item.has_prop(prop):
% for gen, value in sorted(item.iter_prop(prop), reverse=True):
#define ${gen.prefix(item.token_name)}_${prop}  ${value}
% endfor

static inline uint32_t ATTRIBUTE_PURE
${item.token_name}_${prop}(const struct gen_device_info *devinfo)
{
   switch (devinfo->gen) {
   case 11: return ${item.get_prop(prop, 11)};
   case 10: return ${item.get_prop(prop, 10)};
   case 9: return ${item.get_prop(prop, 9)};
   case 8: return ${item.get_prop(prop, 8)};
   case 7:
      if (devinfo->is_haswell) {
         return ${item.get_prop(prop, 7.5)};
      } else {
         return ${item.get_prop(prop, 7)};
      }
   case 6: return ${item.get_prop(prop, 6)};
   case 5: return ${item.get_prop(prop, 5)};
   case 4:
      if (devinfo->is_g4x) {
         return ${item.get_prop(prop, 4.5)};
      } else {
         return ${item.get_prop(prop, 4)};
      }
   default:
      unreachable("Invalid hardware generation");
   }
}
%endif
</%def>

#ifdef __cplusplus
extern "C" {
#endif
% for _, container in sorted(containers.items(), key=itemgetter(0)):

/* ${container.name} */

${emit_per_gen_prop_func(container, 'length')}

% for _, field in sorted(container.fields.items(), key=itemgetter(0)):

/* ${container.name}::${field.name} */

${emit_per_gen_prop_func(field, 'bits')}

${emit_per_gen_prop_func(field, 'start')}

% endfor
% endfor

#ifdef __cplusplus
}
#endif

#endif /* ${guard} */""", output_encoding='utf-8')

def to_alphanum(name):
    substitutions = {
        ' ': '',
        '/': '',
        '[': '',
        ']': '',
        '(': '',
        ')': '',
        '-': '',
        ':': '',
        '.': '',
        ',': '',
        '=': '',
        '>': '',
        '#': '',
        'α': 'alpha',
        '&': '',
        '*': '',
        '"': '',
        '+': '',
        '\'': '',
    }

    for i, j in substitutions.items():
        name = name.replace(i, j)

    return name

def safe_name(name):
    name = to_alphanum(name)
    if not name[0].isalpha():
        name = '_' + name
    return name

class Gen(object):

    def __init__(self, z):
        # Convert potential "major.minor" string
        self.tenx = int(float(z) * 10)

    def __lt__(self, other):
        return self.tenx < other.tenx

    def __hash__(self):
        return hash(self.tenx)

    def __eq__(self, other):
        return self.tenx == other.tenx

    def prefix(self, token):
        gen = self.tenx

        if gen % 10 == 0:
            gen //= 10

        if token[0] == '_':
            token = token[1:]

        return 'GEN{}_{}'.format(gen, token)

class Container(object):

    def __init__(self, name):
        self.name = name
        self.token_name = safe_name(name)
        self.length_by_gen = {}
        self.fields = {}

    def add_gen(self, gen, xml_attrs):
        assert isinstance(gen, Gen)
        if 'length' in xml_attrs:
            self.length_by_gen[gen] = xml_attrs['length']

    def get_field(self, field_name, create=False):
        if field_name not in self.fields:
            if create:
                self.fields[field_name] = Field(self, field_name)
            else:
                return None
        return self.fields[field_name]

    def has_prop(self, prop):
        if prop == 'length':
            return bool(self.length_by_gen)
        else:
            raise ValueError('Invalid property: "{0}"'.format(prop))

    def iter_prop(self, prop):
        if prop == 'length':
            return self.length_by_gen.items()
        else:
            raise ValueError('Invalid property: "{0}"'.format(prop))

    def get_prop(self, prop, gen):
        if not isinstance(gen, Gen):
            gen = Gen(gen)

        if prop == 'length':
            return self.length_by_gen.get(gen, 0)
        else:
            raise ValueError('Invalid property: "{0}"'.format(prop))

class Field(object):

    def __init__(self, container, name):
        self.name = name
        self.token_name = safe_name('_'.join([container.name, self.name]))
        self.bits_by_gen = {}
        self.start_by_gen = {}

    def add_gen(self, gen, xml_attrs):
        assert isinstance(gen, Gen)
        start = int(xml_attrs['start'])
        end = int(xml_attrs['end'])
        self.start_by_gen[gen] = start
        self.bits_by_gen[gen] = 1 + end - start

    def has_prop(self, prop):
        return True

    def iter_prop(self, prop):
        if prop == 'bits':
            return self.bits_by_gen.items()
        elif prop == 'start':
            return self.start_by_gen.items()
        else:
            raise ValueError('Invalid property: "{0}"'.format(prop))

    def get_prop(self, prop, gen):
        if not isinstance(gen, Gen):
            gen = Gen(gen)

        if prop == 'bits':
            return self.bits_by_gen.get(gen, 0)
        elif prop == 'start':
            return self.start_by_gen.get(gen, 0)
        else:
            raise ValueError('Invalid property: "{0}"'.format(prop))

class XmlParser(object):

    def __init__(self, containers):
        self.parser = xml.parsers.expat.ParserCreate()
        self.parser.StartElementHandler = self.start_element
        self.parser.EndElementHandler = self.end_element

        self.gen = None
        self.containers = containers
        self.container = None

    def parse(self, filename):
        with open(filename, 'rb') as f:
            self.parser.ParseFile(f)

    def start_element(self, name, attrs):
        if name == 'genxml':
            self.gen = Gen(attrs['gen'])
        elif name in ('instruction', 'struct', 'register'):
            self.start_container(attrs)
        elif name == 'field':
            self.start_field(attrs)
        else:
            pass

    def end_element(self, name):
        if name == 'genxml':
            self.gen = None
        elif name in ('instruction', 'struct', 'register'):
            self.container = None
        else:
            pass

    def start_container(self, attrs):
        assert self.container is None
        name = attrs['name']
        if name not in self.containers:
            self.containers[name] = Container(name)
        self.container = self.containers[name]
        self.container.add_gen(self.gen, attrs)

    def start_field(self, attrs):
        if self.container is None:
            return

        field_name = attrs.get('name', None)
        if not field_name:
            return

        self.container.get_field(field_name, True).add_gen(self.gen, attrs)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('-o', '--output', type=str,
                   help="If OUTPUT is unset or '-', then it defaults to '/dev/stdout'")
    p.add_argument('--cpp-guard', type=str,
                   help='If unset, then CPP_GUARD is derived from OUTPUT.')
    p.add_argument('xml_sources', metavar='XML_SOURCE', nargs='+')

    pargs = p.parse_args()

    if pargs.output in (None, '-'):
        pargs.output = '/dev/stdout'

    if pargs.cpp_guard is None:
        pargs.cpp_guard = os.path.basename(pargs.output).upper().replace('.', '_')

    return pargs

def main():
    pargs = parse_args()

    # Maps name => Container
    containers = {}

    for source in pargs.xml_sources:
        XmlParser(containers).parse(source)

    with open(pargs.output, 'wb') as f:
        f.write(TEMPLATE.render(containers=containers, guard=pargs.cpp_guard))

if __name__ == '__main__':
    main()
