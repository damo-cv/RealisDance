//
// Copyright 2012 Francisco Jerez
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.
//

#include <sstream>

#include "tgsi/invocation.hpp"
#include "core/error.hpp"

#include "tgsi/tgsi_parse.h"
#include "tgsi/tgsi_text.h"
#include "util/u_memory.h"

using namespace clover;

namespace {
   void
   read_header(const std::string &header, module &m, std::string &r_log) {
      std::istringstream ls(header);
      std::string line;

      while (getline(ls, line)) {
         std::istringstream ts(line);
         std::string name, tok;
         module::size_t offset;
         std::vector<module::argument> args;

         if (!(ts >> name))
            continue;

         if (!(ts >> offset)) {
            r_log = "invalid kernel start address";
            throw build_error();
         }

         while (ts >> tok) {
            if (tok == "scalar")
               args.push_back({ module::argument::scalar, 4 });
            else if (tok == "global")
               args.push_back({ module::argument::global, 4 });
            else if (tok == "local")
               args.push_back({ module::argument::local, 4 });
            else if (tok == "constant")
               args.push_back({ module::argument::constant, 4 });
            else if (tok == "image2d_rd")
               args.push_back({ module::argument::image2d_rd, 4 });
            else if (tok == "image2d_wr")
               args.push_back({ module::argument::image2d_wr, 4 });
            else if (tok == "image3d_rd")
               args.push_back({ module::argument::image3d_rd, 4 });
            else if (tok == "image3d_wr")
               args.push_back({ module::argument::image3d_wr, 4 });
            else if (tok == "sampler")
               args.push_back({ module::argument::sampler, 0 });
            else {
               r_log = "invalid kernel argument";
               throw build_error();
            }
         }

         m.syms.push_back({ name, 0, offset, args });
      }
   }

   void
   read_body(const char *source, module &m, std::string &r_log) {
      tgsi_token prog[1024];

      if (!tgsi_text_translate(source, prog, ARRAY_SIZE(prog))) {
         r_log = "translate failed";
         throw build_error();
      }

      unsigned sz = tgsi_num_tokens(prog) * sizeof(tgsi_token);
      std::vector<char> data( (char *)prog, (char *)prog + sz );
      m.secs.push_back({ 0, module::section::text_executable, sz, data });
   }
}

module
clover::tgsi::compile_program(const std::string &source, std::string &r_log) {
   const size_t body_pos = source.find("COMP\n");
   if (body_pos == std::string::npos) {
      r_log = "invalid source";
      throw build_error();
   }

   const char *body = &source[body_pos];
   module m;

   read_header({ source.begin(), source.begin() + body_pos }, m, r_log);
   read_body(body, m, r_log);

   return m;
}

module
clover::tgsi::link_program(const std::vector<module> &modules)
{
   assert(modules.size() == 1 && "Not implemented");
   return modules[0];
}
