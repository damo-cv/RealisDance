#!/bin/sh

if [ -z "$srcdir" -o -z "$abs_builddir" ]; then
    echo ""
    echo "Warning: you're invoking the script manually and things may fail."
    echo "Attempting to determine/set srcdir and abs_builddir variables."
    echo ""

    # Should point to `dirname Makefile.glsl.am`
    srcdir=./../../../
    cd `dirname "$0"`
    # Should point to `dirname Makefile` equivalent to the above.
    abs_builddir=`pwd`/../../../
fi

$PYTHON $srcdir/glsl/glcpp/tests/glcpp_test.py $abs_builddir/glsl/glcpp/glcpp $srcdir/glsl/glcpp/tests --unix --windows --oldmac --bizarro
