/* A Bison parser, made by GNU Bison 3.1.  */

/* Bison implementation for Yacc-like parsers in C

   Copyright (C) 1984, 1989-1990, 2000-2015, 2018 Free Software Foundation, Inc.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.

   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */

/* C LALR(1) parser skeleton written by Richard Stallman, by
   simplifying the original so-called "semantic" parser.  */

/* All symbols defined below should begin with yy or YY, to avoid
   infringing on user name space.  This should be done even for local
   variables, as they might otherwise be expanded by user macros.
   There are some unavoidable exceptions within include files to
   define necessary library symbols; they are noted "INFRINGES ON
   USER NAME SPACE" below.  */

/* Identify Bison output.  */
#define YYBISON 1

/* Bison version.  */
#define YYBISON_VERSION "3.1"

/* Skeleton name.  */
#define YYSKELETON_NAME "yacc.c"

/* Pure parsers.  */
#define YYPURE 1

/* Push parsers.  */
#define YYPUSH 0

/* Pull parsers.  */
#define YYPULL 1


/* Substitute the variable and function names.  */
#define yyparse         _mesa_glsl_parse
#define yylex           _mesa_glsl_lex
#define yyerror         _mesa_glsl_error
#define yydebug         _mesa_glsl_debug
#define yynerrs         _mesa_glsl_nerrs


/* Copy the first part of user declarations.  */
#line 1 "./glsl/glsl_parser.yy" /* yacc.c:339  */

/*
 * Copyright © 2008, 2009 Intel Corporation
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
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef _MSC_VER
#include <strings.h>
#endif
#include <assert.h>

#include "ast.h"
#include "glsl_parser_extras.h"
#include "compiler/glsl_types.h"
#include "main/context.h"

#ifdef _MSC_VER
#pragma warning( disable : 4065 ) // switch statement contains 'default' but no 'case' labels
#endif

#undef yyerror

static void yyerror(YYLTYPE *loc, _mesa_glsl_parse_state *st, const char *msg)
{
   _mesa_glsl_error(loc, st, "%s", msg);
}

static int
_mesa_glsl_lex(YYSTYPE *val, YYLTYPE *loc, _mesa_glsl_parse_state *state)
{
   return _mesa_glsl_lexer_lex(val, loc, state->scanner);
}

static bool match_layout_qualifier(const char *s1, const char *s2,
                                   _mesa_glsl_parse_state *state)
{
   /* From the GLSL 1.50 spec, section 4.3.8 (Layout Qualifiers):
    *
    *     "The tokens in any layout-qualifier-id-list ... are not case
    *     sensitive, unless explicitly noted otherwise."
    *
    * The text "unless explicitly noted otherwise" appears to be
    * vacuous--no desktop GLSL spec (up through GLSL 4.40) notes
    * otherwise.
    *
    * However, the GLSL ES 3.00 spec says, in section 4.3.8 (Layout
    * Qualifiers):
    *
    *     "As for other identifiers, they are case sensitive."
    *
    * So we need to do a case-sensitive or a case-insensitive match,
    * depending on whether we are compiling for GLSL ES.
    */
   if (state->es_shader)
      return strcmp(s1, s2);
   else
      return strcasecmp(s1, s2);
}

#line 152 "glsl/glsl_parser.cpp" /* yacc.c:339  */

# ifndef YY_NULLPTR
#  if defined __cplusplus && 201103L <= __cplusplus
#   define YY_NULLPTR nullptr
#  else
#   define YY_NULLPTR 0
#  endif
# endif

/* Enabling verbose error messages.  */
#ifdef YYERROR_VERBOSE
# undef YYERROR_VERBOSE
# define YYERROR_VERBOSE 1
#else
# define YYERROR_VERBOSE 1
#endif

/* In a future release of Bison, this section will be replaced
   by #include "glsl_parser.h".  */
#ifndef YY__MESA_GLSL_GLSL_GLSL_PARSER_H_INCLUDED
# define YY__MESA_GLSL_GLSL_GLSL_PARSER_H_INCLUDED
/* Debug traces.  */
#ifndef YYDEBUG
# define YYDEBUG 0
#endif
#if YYDEBUG
extern int _mesa_glsl_debug;
#endif

/* Token type.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
  enum yytokentype
  {
    ATTRIBUTE = 258,
    CONST_TOK = 259,
    BASIC_TYPE_TOK = 260,
    BREAK = 261,
    BUFFER = 262,
    CONTINUE = 263,
    DO = 264,
    ELSE = 265,
    FOR = 266,
    IF = 267,
    DISCARD = 268,
    RETURN = 269,
    SWITCH = 270,
    CASE = 271,
    DEFAULT = 272,
    CENTROID = 273,
    IN_TOK = 274,
    OUT_TOK = 275,
    INOUT_TOK = 276,
    UNIFORM = 277,
    VARYING = 278,
    SAMPLE = 279,
    NOPERSPECTIVE = 280,
    FLAT = 281,
    SMOOTH = 282,
    IMAGE1DSHADOW = 283,
    IMAGE2DSHADOW = 284,
    IMAGE1DARRAYSHADOW = 285,
    IMAGE2DARRAYSHADOW = 286,
    COHERENT = 287,
    VOLATILE = 288,
    RESTRICT = 289,
    READONLY = 290,
    WRITEONLY = 291,
    SHARED = 292,
    STRUCT = 293,
    VOID_TOK = 294,
    WHILE = 295,
    IDENTIFIER = 296,
    TYPE_IDENTIFIER = 297,
    NEW_IDENTIFIER = 298,
    FLOATCONSTANT = 299,
    DOUBLECONSTANT = 300,
    INTCONSTANT = 301,
    UINTCONSTANT = 302,
    BOOLCONSTANT = 303,
    INT64CONSTANT = 304,
    UINT64CONSTANT = 305,
    FIELD_SELECTION = 306,
    LEFT_OP = 307,
    RIGHT_OP = 308,
    INC_OP = 309,
    DEC_OP = 310,
    LE_OP = 311,
    GE_OP = 312,
    EQ_OP = 313,
    NE_OP = 314,
    AND_OP = 315,
    OR_OP = 316,
    XOR_OP = 317,
    MUL_ASSIGN = 318,
    DIV_ASSIGN = 319,
    ADD_ASSIGN = 320,
    MOD_ASSIGN = 321,
    LEFT_ASSIGN = 322,
    RIGHT_ASSIGN = 323,
    AND_ASSIGN = 324,
    XOR_ASSIGN = 325,
    OR_ASSIGN = 326,
    SUB_ASSIGN = 327,
    INVARIANT = 328,
    PRECISE = 329,
    LOWP = 330,
    MEDIUMP = 331,
    HIGHP = 332,
    SUPERP = 333,
    PRECISION = 334,
    VERSION_TOK = 335,
    EXTENSION = 336,
    LINE = 337,
    COLON = 338,
    EOL = 339,
    INTERFACE = 340,
    OUTPUT = 341,
    PRAGMA_DEBUG_ON = 342,
    PRAGMA_DEBUG_OFF = 343,
    PRAGMA_OPTIMIZE_ON = 344,
    PRAGMA_OPTIMIZE_OFF = 345,
    PRAGMA_INVARIANT_ALL = 346,
    LAYOUT_TOK = 347,
    DOT_TOK = 348,
    ASM = 349,
    CLASS = 350,
    UNION = 351,
    ENUM = 352,
    TYPEDEF = 353,
    TEMPLATE = 354,
    THIS = 355,
    PACKED_TOK = 356,
    GOTO = 357,
    INLINE_TOK = 358,
    NOINLINE = 359,
    PUBLIC_TOK = 360,
    STATIC = 361,
    EXTERN = 362,
    EXTERNAL = 363,
    LONG_TOK = 364,
    SHORT_TOK = 365,
    HALF = 366,
    FIXED_TOK = 367,
    UNSIGNED = 368,
    INPUT_TOK = 369,
    HVEC2 = 370,
    HVEC3 = 371,
    HVEC4 = 372,
    FVEC2 = 373,
    FVEC3 = 374,
    FVEC4 = 375,
    SAMPLER3DRECT = 376,
    SIZEOF = 377,
    CAST = 378,
    NAMESPACE = 379,
    USING = 380,
    RESOURCE = 381,
    PATCH = 382,
    SUBROUTINE = 383,
    ERROR_TOK = 384,
    COMMON = 385,
    PARTITION = 386,
    ACTIVE = 387,
    FILTER = 388,
    ROW_MAJOR = 389,
    THEN = 390
  };
#endif

/* Value type.  */
#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED

union YYSTYPE
{
#line 98 "./glsl/glsl_parser.yy" /* yacc.c:355  */

   int n;
   int64_t n64;
   float real;
   double dreal;
   const char *identifier;

   struct ast_type_qualifier type_qualifier;

   ast_node *node;
   ast_type_specifier *type_specifier;
   ast_array_specifier *array_specifier;
   ast_fully_specified_type *fully_specified_type;
   ast_function *function;
   ast_parameter_declarator *parameter_declarator;
   ast_function_definition *function_definition;
   ast_compound_statement *compound_statement;
   ast_expression *expression;
   ast_declarator_list *declarator_list;
   ast_struct_specifier *struct_specifier;
   ast_declaration *declaration;
   ast_switch_body *switch_body;
   ast_case_label *case_label;
   ast_case_label_list *case_label_list;
   ast_case_statement *case_statement;
   ast_case_statement_list *case_statement_list;
   ast_interface_block *interface_block;
   ast_subroutine_list *subroutine_list;
   struct {
      ast_node *cond;
      ast_expression *rest;
   } for_rest_statement;

   struct {
      ast_node *then_statement;
      ast_node *else_statement;
   } selection_rest_statement;

   const glsl_type *type;

#line 369 "glsl/glsl_parser.cpp" /* yacc.c:355  */
};

typedef union YYSTYPE YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define YYSTYPE_IS_DECLARED 1
#endif

/* Location type.  */
#if ! defined YYLTYPE && ! defined YYLTYPE_IS_DECLARED
typedef struct YYLTYPE YYLTYPE;
struct YYLTYPE
{
  int first_line;
  int first_column;
  int last_line;
  int last_column;
};
# define YYLTYPE_IS_DECLARED 1
# define YYLTYPE_IS_TRIVIAL 1
#endif



int _mesa_glsl_parse (struct _mesa_glsl_parse_state *state);

#endif /* !YY__MESA_GLSL_GLSL_GLSL_PARSER_H_INCLUDED  */

/* Copy the second part of user declarations.  */

#line 399 "glsl/glsl_parser.cpp" /* yacc.c:358  */

#ifdef short
# undef short
#endif

#ifdef YYTYPE_UINT8
typedef YYTYPE_UINT8 yytype_uint8;
#else
typedef unsigned char yytype_uint8;
#endif

#ifdef YYTYPE_INT8
typedef YYTYPE_INT8 yytype_int8;
#else
typedef signed char yytype_int8;
#endif

#ifdef YYTYPE_UINT16
typedef YYTYPE_UINT16 yytype_uint16;
#else
typedef unsigned short yytype_uint16;
#endif

#ifdef YYTYPE_INT16
typedef YYTYPE_INT16 yytype_int16;
#else
typedef short yytype_int16;
#endif

#ifndef YYSIZE_T
# ifdef __SIZE_TYPE__
#  define YYSIZE_T __SIZE_TYPE__
# elif defined size_t
#  define YYSIZE_T size_t
# elif ! defined YYSIZE_T
#  include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  define YYSIZE_T size_t
# else
#  define YYSIZE_T unsigned
# endif
#endif

#define YYSIZE_MAXIMUM ((YYSIZE_T) -1)

#ifndef YY_
# if defined YYENABLE_NLS && YYENABLE_NLS
#  if ENABLE_NLS
#   include <libintl.h> /* INFRINGES ON USER NAME SPACE */
#   define YY_(Msgid) dgettext ("bison-runtime", Msgid)
#  endif
# endif
# ifndef YY_
#  define YY_(Msgid) Msgid
# endif
#endif

#ifndef YY_ATTRIBUTE
# if (defined __GNUC__                                               \
      && (2 < __GNUC__ || (__GNUC__ == 2 && 96 <= __GNUC_MINOR__)))  \
     || defined __SUNPRO_C && 0x5110 <= __SUNPRO_C
#  define YY_ATTRIBUTE(Spec) __attribute__(Spec)
# else
#  define YY_ATTRIBUTE(Spec) /* empty */
# endif
#endif

#ifndef YY_ATTRIBUTE_PURE
# define YY_ATTRIBUTE_PURE   YY_ATTRIBUTE ((__pure__))
#endif

#ifndef YY_ATTRIBUTE_UNUSED
# define YY_ATTRIBUTE_UNUSED YY_ATTRIBUTE ((__unused__))
#endif

#if !defined _Noreturn \
     && (!defined __STDC_VERSION__ || __STDC_VERSION__ < 201112)
# if defined _MSC_VER && 1200 <= _MSC_VER
#  define _Noreturn __declspec (noreturn)
# else
#  define _Noreturn YY_ATTRIBUTE ((__noreturn__))
# endif
#endif

/* Suppress unused-variable warnings by "using" E.  */
#if ! defined lint || defined __GNUC__
# define YYUSE(E) ((void) (E))
#else
# define YYUSE(E) /* empty */
#endif

#if defined __GNUC__ && ! defined __ICC && 407 <= __GNUC__ * 100 + __GNUC_MINOR__
/* Suppress an incorrect diagnostic about yylval being uninitialized.  */
# define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN \
    _Pragma ("GCC diagnostic push") \
    _Pragma ("GCC diagnostic ignored \"-Wuninitialized\"")\
    _Pragma ("GCC diagnostic ignored \"-Wmaybe-uninitialized\"")
# define YY_IGNORE_MAYBE_UNINITIALIZED_END \
    _Pragma ("GCC diagnostic pop")
#else
# define YY_INITIAL_VALUE(Value) Value
#endif
#ifndef YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_END
#endif
#ifndef YY_INITIAL_VALUE
# define YY_INITIAL_VALUE(Value) /* Nothing. */
#endif


#if ! defined yyoverflow || YYERROR_VERBOSE

/* The parser invokes alloca or malloc; define the necessary symbols.  */

# ifdef YYSTACK_USE_ALLOCA
#  if YYSTACK_USE_ALLOCA
#   ifdef __GNUC__
#    define YYSTACK_ALLOC __builtin_alloca
#   elif defined __BUILTIN_VA_ARG_INCR
#    include <alloca.h> /* INFRINGES ON USER NAME SPACE */
#   elif defined _AIX
#    define YYSTACK_ALLOC __alloca
#   elif defined _MSC_VER
#    include <malloc.h> /* INFRINGES ON USER NAME SPACE */
#    define alloca _alloca
#   else
#    define YYSTACK_ALLOC alloca
#    if ! defined _ALLOCA_H && ! defined EXIT_SUCCESS
#     include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
      /* Use EXIT_SUCCESS as a witness for stdlib.h.  */
#     ifndef EXIT_SUCCESS
#      define EXIT_SUCCESS 0
#     endif
#    endif
#   endif
#  endif
# endif

# ifdef YYSTACK_ALLOC
   /* Pacify GCC's 'empty if-body' warning.  */
#  define YYSTACK_FREE(Ptr) do { /* empty */; } while (0)
#  ifndef YYSTACK_ALLOC_MAXIMUM
    /* The OS might guarantee only one guard page at the bottom of the stack,
       and a page size can be as small as 4096 bytes.  So we cannot safely
       invoke alloca (N) if N exceeds 4096.  Use a slightly smaller number
       to allow for a few compiler-allocated temporary stack slots.  */
#   define YYSTACK_ALLOC_MAXIMUM 4032 /* reasonable circa 2006 */
#  endif
# else
#  define YYSTACK_ALLOC YYMALLOC
#  define YYSTACK_FREE YYFREE
#  ifndef YYSTACK_ALLOC_MAXIMUM
#   define YYSTACK_ALLOC_MAXIMUM YYSIZE_MAXIMUM
#  endif
#  if (defined __cplusplus && ! defined EXIT_SUCCESS \
       && ! ((defined YYMALLOC || defined malloc) \
             && (defined YYFREE || defined free)))
#   include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#   ifndef EXIT_SUCCESS
#    define EXIT_SUCCESS 0
#   endif
#  endif
#  ifndef YYMALLOC
#   define YYMALLOC malloc
#   if ! defined malloc && ! defined EXIT_SUCCESS
void *malloc (YYSIZE_T); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
#  ifndef YYFREE
#   define YYFREE free
#   if ! defined free && ! defined EXIT_SUCCESS
void free (void *); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
# endif
#endif /* ! defined yyoverflow || YYERROR_VERBOSE */


#if (! defined yyoverflow \
     && (! defined __cplusplus \
         || (defined YYLTYPE_IS_TRIVIAL && YYLTYPE_IS_TRIVIAL \
             && defined YYSTYPE_IS_TRIVIAL && YYSTYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union yyalloc
{
  yytype_int16 yyss_alloc;
  YYSTYPE yyvs_alloc;
  YYLTYPE yyls_alloc;
};

/* The size of the maximum gap between one aligned stack and the next.  */
# define YYSTACK_GAP_MAXIMUM (sizeof (union yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define YYSTACK_BYTES(N) \
     ((N) * (sizeof (yytype_int16) + sizeof (YYSTYPE) + sizeof (YYLTYPE)) \
      + 2 * YYSTACK_GAP_MAXIMUM)

# define YYCOPY_NEEDED 1

/* Relocate STACK from its old location to the new one.  The
   local variables YYSIZE and YYSTACKSIZE give the old and new number of
   elements in the stack, and YYPTR gives the new location of the
   stack.  Advance YYPTR to a properly aligned location for the next
   stack.  */
# define YYSTACK_RELOCATE(Stack_alloc, Stack)                           \
    do                                                                  \
      {                                                                 \
        YYSIZE_T yynewbytes;                                            \
        YYCOPY (&yyptr->Stack_alloc, Stack, yysize);                    \
        Stack = &yyptr->Stack_alloc;                                    \
        yynewbytes = yystacksize * sizeof (*Stack) + YYSTACK_GAP_MAXIMUM; \
        yyptr += yynewbytes / sizeof (*yyptr);                          \
      }                                                                 \
    while (0)

#endif

#if defined YYCOPY_NEEDED && YYCOPY_NEEDED
/* Copy COUNT objects from SRC to DST.  The source and destination do
   not overlap.  */
# ifndef YYCOPY
#  if defined __GNUC__ && 1 < __GNUC__
#   define YYCOPY(Dst, Src, Count) \
      __builtin_memcpy (Dst, Src, (Count) * sizeof (*(Src)))
#  else
#   define YYCOPY(Dst, Src, Count)              \
      do                                        \
        {                                       \
          YYSIZE_T yyi;                         \
          for (yyi = 0; yyi < (Count); yyi++)   \
            (Dst)[yyi] = (Src)[yyi];            \
        }                                       \
      while (0)
#  endif
# endif
#endif /* !YYCOPY_NEEDED */

/* YYFINAL -- State number of the termination state.  */
#define YYFINAL  5
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   2286

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  159
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  110
/* YYNRULES -- Number of rules.  */
#define YYNRULES  306
/* YYNSTATES -- Number of states.  */
#define YYNSTATES  465

/* YYTRANSLATE[YYX] -- Symbol number corresponding to YYX as returned
   by yylex, with out-of-bounds checking.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   390

#define YYTRANSLATE(YYX)                                                \
  ((unsigned) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[TOKEN-NUM] -- Symbol number corresponding to TOKEN-NUM
   as returned by yylex, without out-of-bounds checking.  */
static const yytype_uint8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,   143,     2,     2,     2,   147,   150,     2,
     136,   137,   145,   141,   140,   142,     2,   146,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,   154,   156,
     148,   155,   149,   153,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,   138,     2,   139,   151,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,   157,   152,   158,   144,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     1,     2,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    37,    38,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    50,    51,    52,    53,    54,
      55,    56,    57,    58,    59,    60,    61,    62,    63,    64,
      65,    66,    67,    68,    69,    70,    71,    72,    73,    74,
      75,    76,    77,    78,    79,    80,    81,    82,    83,    84,
      85,    86,    87,    88,    89,    90,    91,    92,    93,    94,
      95,    96,    97,    98,    99,   100,   101,   102,   103,   104,
     105,   106,   107,   108,   109,   110,   111,   112,   113,   114,
     115,   116,   117,   118,   119,   120,   121,   122,   123,   124,
     125,   126,   127,   128,   129,   130,   131,   132,   133,   134,
     135
};

#if YYDEBUG
  /* YYRLINE[YYN] -- Source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,   289,   289,   288,   312,   314,   321,   331,   332,   333,
     334,   335,   359,   361,   365,   366,   367,   371,   380,   388,
     396,   407,   408,   412,   419,   426,   433,   440,   447,   454,
     461,   468,   475,   476,   482,   486,   493,   499,   508,   512,
     516,   520,   521,   525,   526,   530,   536,   548,   552,   558,
     572,   573,   579,   585,   595,   596,   597,   598,   602,   603,
     609,   615,   624,   625,   631,   640,   641,   647,   656,   657,
     663,   669,   675,   684,   685,   691,   700,   701,   710,   711,
     720,   721,   730,   731,   740,   741,   750,   751,   760,   761,
     770,   771,   780,   781,   782,   783,   784,   785,   786,   787,
     788,   789,   790,   794,   798,   814,   818,   823,   827,   832,
     849,   853,   854,   858,   863,   871,   889,   900,   917,   932,
     940,   957,   960,   968,   976,   988,  1000,  1007,  1012,  1017,
    1026,  1030,  1031,  1041,  1051,  1061,  1075,  1082,  1093,  1104,
    1115,  1126,  1138,  1153,  1160,  1178,  1185,  1186,  1196,  1650,
    1815,  1841,  1846,  1851,  1859,  1864,  1873,  1882,  1894,  1899,
    1904,  1913,  1918,  1923,  1924,  1925,  1926,  1927,  1928,  1929,
    1947,  1955,  1980,  2004,  2018,  2023,  2039,  2059,  2071,  2079,
    2084,  2089,  2096,  2101,  2106,  2111,  2116,  2141,  2153,  2158,
    2163,  2171,  2176,  2181,  2187,  2192,  2200,  2208,  2214,  2224,
    2235,  2236,  2244,  2250,  2256,  2265,  2266,  2270,  2275,  2280,
    2288,  2295,  2312,  2317,  2325,  2363,  2368,  2376,  2382,  2391,
    2392,  2396,  2403,  2410,  2417,  2423,  2424,  2428,  2429,  2430,
    2431,  2432,  2433,  2437,  2444,  2443,  2457,  2458,  2462,  2468,
    2477,  2487,  2499,  2505,  2514,  2523,  2528,  2536,  2540,  2558,
    2566,  2571,  2579,  2584,  2592,  2600,  2608,  2616,  2624,  2632,
    2640,  2647,  2654,  2664,  2665,  2669,  2671,  2677,  2682,  2691,
    2697,  2703,  2709,  2715,  2724,  2725,  2726,  2727,  2728,  2732,
    2746,  2750,  2763,  2781,  2800,  2805,  2810,  2815,  2820,  2835,
    2838,  2843,  2851,  2856,  2864,  2888,  2895,  2899,  2906,  2910,
    2920,  2929,  2939,  2948,  2960,  2982,  2992
};
#endif

#if YYDEBUG || YYERROR_VERBOSE || 1
/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "$end", "error", "$undefined", "ATTRIBUTE", "CONST_TOK",
  "BASIC_TYPE_TOK", "BREAK", "BUFFER", "CONTINUE", "DO", "ELSE", "FOR",
  "IF", "DISCARD", "RETURN", "SWITCH", "CASE", "DEFAULT", "CENTROID",
  "IN_TOK", "OUT_TOK", "INOUT_TOK", "UNIFORM", "VARYING", "SAMPLE",
  "NOPERSPECTIVE", "FLAT", "SMOOTH", "IMAGE1DSHADOW", "IMAGE2DSHADOW",
  "IMAGE1DARRAYSHADOW", "IMAGE2DARRAYSHADOW", "COHERENT", "VOLATILE",
  "RESTRICT", "READONLY", "WRITEONLY", "SHARED", "STRUCT", "VOID_TOK",
  "WHILE", "IDENTIFIER", "TYPE_IDENTIFIER", "NEW_IDENTIFIER",
  "FLOATCONSTANT", "DOUBLECONSTANT", "INTCONSTANT", "UINTCONSTANT",
  "BOOLCONSTANT", "INT64CONSTANT", "UINT64CONSTANT", "FIELD_SELECTION",
  "LEFT_OP", "RIGHT_OP", "INC_OP", "DEC_OP", "LE_OP", "GE_OP", "EQ_OP",
  "NE_OP", "AND_OP", "OR_OP", "XOR_OP", "MUL_ASSIGN", "DIV_ASSIGN",
  "ADD_ASSIGN", "MOD_ASSIGN", "LEFT_ASSIGN", "RIGHT_ASSIGN", "AND_ASSIGN",
  "XOR_ASSIGN", "OR_ASSIGN", "SUB_ASSIGN", "INVARIANT", "PRECISE", "LOWP",
  "MEDIUMP", "HIGHP", "SUPERP", "PRECISION", "VERSION_TOK", "EXTENSION",
  "LINE", "COLON", "EOL", "INTERFACE", "OUTPUT", "PRAGMA_DEBUG_ON",
  "PRAGMA_DEBUG_OFF", "PRAGMA_OPTIMIZE_ON", "PRAGMA_OPTIMIZE_OFF",
  "PRAGMA_INVARIANT_ALL", "LAYOUT_TOK", "DOT_TOK", "ASM", "CLASS", "UNION",
  "ENUM", "TYPEDEF", "TEMPLATE", "THIS", "PACKED_TOK", "GOTO",
  "INLINE_TOK", "NOINLINE", "PUBLIC_TOK", "STATIC", "EXTERN", "EXTERNAL",
  "LONG_TOK", "SHORT_TOK", "HALF", "FIXED_TOK", "UNSIGNED", "INPUT_TOK",
  "HVEC2", "HVEC3", "HVEC4", "FVEC2", "FVEC3", "FVEC4", "SAMPLER3DRECT",
  "SIZEOF", "CAST", "NAMESPACE", "USING", "RESOURCE", "PATCH",
  "SUBROUTINE", "ERROR_TOK", "COMMON", "PARTITION", "ACTIVE", "FILTER",
  "ROW_MAJOR", "THEN", "'('", "')'", "'['", "']'", "','", "'+'", "'-'",
  "'!'", "'~'", "'*'", "'/'", "'%'", "'<'", "'>'", "'&'", "'^'", "'|'",
  "'?'", "':'", "'='", "';'", "'{'", "'}'", "$accept", "translation_unit",
  "$@1", "version_statement", "pragma_statement",
  "extension_statement_list", "any_identifier", "extension_statement",
  "external_declaration_list", "variable_identifier", "primary_expression",
  "postfix_expression", "integer_expression", "function_call",
  "function_call_or_method", "function_call_generic",
  "function_call_header_no_parameters",
  "function_call_header_with_parameters", "function_call_header",
  "function_identifier", "unary_expression", "unary_operator",
  "multiplicative_expression", "additive_expression", "shift_expression",
  "relational_expression", "equality_expression", "and_expression",
  "exclusive_or_expression", "inclusive_or_expression",
  "logical_and_expression", "logical_xor_expression",
  "logical_or_expression", "conditional_expression",
  "assignment_expression", "assignment_operator", "expression",
  "constant_expression", "declaration", "function_prototype",
  "function_declarator", "function_header_with_parameters",
  "function_header", "parameter_declarator", "parameter_declaration",
  "parameter_qualifier", "parameter_direction_qualifier",
  "parameter_type_specifier", "init_declarator_list", "single_declaration",
  "fully_specified_type", "layout_qualifier", "layout_qualifier_id_list",
  "layout_qualifier_id", "interface_block_layout_qualifier",
  "subroutine_qualifier", "subroutine_type_list",
  "interpolation_qualifier", "type_qualifier",
  "auxiliary_storage_qualifier", "storage_qualifier", "memory_qualifier",
  "array_specifier", "type_specifier", "type_specifier_nonarray",
  "basic_type_specifier_nonarray", "precision_qualifier",
  "struct_specifier", "struct_declaration_list", "struct_declaration",
  "struct_declarator_list", "struct_declarator", "initializer",
  "initializer_list", "declaration_statement", "statement",
  "simple_statement", "compound_statement", "$@2",
  "statement_no_new_scope", "compound_statement_no_new_scope",
  "statement_list", "expression_statement", "selection_statement",
  "selection_rest_statement", "condition", "switch_statement",
  "switch_body", "case_label", "case_label_list", "case_statement",
  "case_statement_list", "iteration_statement", "for_init_statement",
  "conditionopt", "for_rest_statement", "jump_statement",
  "external_declaration", "function_definition", "interface_block",
  "basic_interface_block", "interface_qualifier", "instance_name_opt",
  "member_list", "member_declaration", "layout_uniform_defaults",
  "layout_buffer_defaults", "layout_in_defaults", "layout_out_defaults",
  "layout_defaults", YY_NULLPTR
};
#endif

# ifdef YYPRINT
/* YYTOKNUM[NUM] -- (External) token number corresponding to the
   (internal) symbol number NUM (which must be that of a token).  */
static const yytype_uint16 yytoknum[] =
{
       0,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   278,   279,   280,   281,   282,   283,   284,
     285,   286,   287,   288,   289,   290,   291,   292,   293,   294,
     295,   296,   297,   298,   299,   300,   301,   302,   303,   304,
     305,   306,   307,   308,   309,   310,   311,   312,   313,   314,
     315,   316,   317,   318,   319,   320,   321,   322,   323,   324,
     325,   326,   327,   328,   329,   330,   331,   332,   333,   334,
     335,   336,   337,   338,   339,   340,   341,   342,   343,   344,
     345,   346,   347,   348,   349,   350,   351,   352,   353,   354,
     355,   356,   357,   358,   359,   360,   361,   362,   363,   364,
     365,   366,   367,   368,   369,   370,   371,   372,   373,   374,
     375,   376,   377,   378,   379,   380,   381,   382,   383,   384,
     385,   386,   387,   388,   389,   390,    40,    41,    91,    93,
      44,    43,    45,    33,   126,    42,    47,    37,    60,    62,
      38,    94,   124,    63,    58,    61,    59,   123,   125
};
# endif

#define YYPACT_NINF -278

#define yypact_value_is_default(Yystate) \
  (!!((Yystate) == (-278)))

#define YYTABLE_NINF -288

#define yytable_value_is_error(Yytable_value) \
  0

  /* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
     STATE-NUM.  */
static const yytype_int16 yypact[] =
{
     -48,   -44,    34,  -278,    -1,  -278,   -16,  -278,  -278,  -278,
    -278,   -45,   180,  1677,  -278,  -278,   -25,  -278,  -278,  -278,
      26,  -278,    61,    86,  -278,   104,  -278,  -278,  -278,  -278,
    -278,  -278,  -278,  -278,  -278,  -278,  -278,   -26,  -278,  -278,
    1975,  1975,  -278,  -278,  -278,   163,    13,    75,    77,    95,
     106,    32,  -278,    56,  -278,  -278,  1580,  -278,    53,    66,
      58,   309,  -102,  -278,   221,  2036,  2097,  2097,    48,  2158,
    2097,  2158,  -278,    78,  -278,  2097,  -278,  -278,  -278,  -278,
    -278,   176,  -278,  -278,  -278,  -278,  -278,   180,  1900,    71,
    -278,  -278,  -278,  -278,  -278,  -278,  2097,  2097,  -278,  2097,
    -278,  2097,  2097,  -278,  -278,    48,  -278,  -278,  -278,  -278,
    -278,    -6,   180,  -278,  -278,  -278,   500,  -278,  -278,   563,
     563,  -278,  -278,  -278,   563,  -278,    43,   563,   563,   563,
     180,  -278,    94,    99,  -111,   107,   -32,   -24,   -23,   -22,
    -278,  -278,  -278,  -278,  -278,  -278,  -278,  -278,  -278,  -278,
    -278,  -278,  2158,  -278,  -278,   576,   120,  -278,   103,   185,
     180,   889,  -278,  1900,   121,  -278,  -278,  -278,   131,  -107,
    -278,  -278,  -278,   -38,   135,   139,  1220,   151,   158,   140,
    1430,   164,   165,  -278,  -278,  -278,  -278,  -278,  -278,  -278,
    1140,  1140,  1140,  -278,  -278,  -278,  -278,  -278,   144,  -278,
    -278,  -278,   134,  -278,  -278,  -278,   166,    29,  1295,   168,
     251,  1140,   138,    84,   196,     7,   174,   155,   157,   154,
     249,   250,   -53,  -278,  -278,   -42,  -278,   171,   175,  -278,
    -278,  -278,  -278,   656,  -278,  -278,  -278,  -278,  -278,  -278,
    -278,  -278,  -278,  -278,    48,   180,  -278,  -278,  -278,  -109,
     980,   -78,  -278,  -278,  -278,  -278,  -278,  -278,  -278,  -278,
     186,  -278,   732,  1900,  -278,    78,   -17,  -278,  -278,  -278,
     968,  -278,  1140,  -278,    -6,  -278,   180,  -278,  -278,   291,
    1502,  1140,  -278,  -278,     2,  1140,  1825,  -278,  -278,    54,
    -278,  1220,  -278,  -278,   281,  1140,  -278,  -278,  1140,   197,
    -278,  -278,  -278,  -278,  -278,  -278,  -278,  -278,  -278,  -278,
    -278,  -278,  -278,  1140,  -278,  1140,  1140,  1140,  1140,  1140,
    1140,  1140,  1140,  1140,  1140,  1140,  1140,  1140,  1140,  1140,
    1140,  1140,  1140,  1140,  1140,  1140,  -278,  -278,  -278,   180,
      78,   980,   -50,   980,  -278,  -278,   980,  -278,  -278,   194,
     180,   177,  1900,   120,   180,  -278,  -278,  -278,  -278,  -278,
     200,  -278,  -278,  1825,    60,  -278,    68,   198,   180,   203,
    -278,   812,  -278,   207,   198,  -278,  -278,  -278,  -278,  -278,
     138,   138,    84,    84,   196,   196,   196,   196,     7,     7,
     174,   155,   157,   154,   249,   250,   -70,  -278,  -278,   120,
    -278,   980,  -278,   -79,  -278,  -278,    17,   294,  -278,  -278,
    1140,  -278,   191,   211,  1220,   192,   199,  1375,  -278,  -278,
    1140,  -278,  1683,  -278,  -278,    78,   195,    74,  1140,  1375,
     342,  -278,    -7,  -278,   980,  -278,  -278,  -278,  -278,  -278,
    -278,   120,  -278,   201,   198,  -278,  1220,  1140,   202,  -278,
    -278,  1065,  1220,    -4,  -278,  -278,  -278,    41,  -278,  -278,
    -278,  -278,  -278,  1220,  -278
};

  /* YYDEFACT[STATE-NUM] -- Default reduction number in state STATE-NUM.
     Performed when YYTABLE does not specify something else to do.  Zero
     means the default is an error.  */
static const yytype_uint16 yydefact[] =
{
       4,     0,     0,    12,     0,     1,     2,    14,    15,    16,
       5,     0,     0,     0,    13,     6,     0,   183,   182,   206,
     189,   179,   185,   186,   187,   188,   184,   180,   160,   159,
     158,   191,   192,   193,   194,   195,   190,     0,   205,   204,
     161,   162,   209,   208,   207,     0,     0,     0,     0,     0,
       0,     0,   181,   154,   278,   276,     3,   275,     0,     0,
     112,   121,     0,   131,   136,   166,   168,   165,     0,   163,
     164,   167,   143,   200,   202,   169,   203,    18,   274,   109,
     280,     0,   303,   304,   305,   306,   277,     0,     0,     0,
     189,   185,   186,   188,    21,    22,   161,   162,   141,   166,
     171,   163,   167,   142,   170,     0,     7,     8,     9,    10,
      11,     0,     0,    20,    19,   106,     0,   279,   110,   121,
     121,   127,   128,   129,   121,   113,     0,   121,   121,   121,
       0,   107,    14,    16,   137,     0,   189,   185,   186,   188,
     173,   281,   295,   297,   299,   301,   174,   172,   144,   175,
     288,   176,   166,   178,   282,     0,   201,   177,     0,     0,
       0,     0,   212,     0,     0,   153,   152,   151,   148,     0,
     146,   150,   156,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    28,    29,    24,    25,    30,    26,    27,
       0,     0,     0,    54,    55,    56,    57,   242,   234,   238,
      23,    32,    50,    34,    39,    40,     0,     0,    44,     0,
      58,     0,    62,    65,    68,    73,    76,    78,    80,    82,
      84,    86,    88,    90,   103,     0,   224,     0,   143,   227,
     240,   226,   225,     0,   228,   229,   230,   231,   232,   114,
     122,   123,   119,   120,     0,   130,   124,   126,   125,   132,
       0,   138,   115,   298,   300,   302,   296,   196,    58,   105,
       0,    48,     0,     0,    17,   217,     0,   215,   211,   213,
       0,   108,     0,   145,     0,   155,     0,   270,   269,     0,
       0,     0,   273,   271,     0,     0,     0,    51,    52,     0,
     233,     0,    36,    37,     0,     0,    42,    41,     0,   205,
      45,    47,    93,    94,    96,    95,    98,    99,   100,   101,
     102,    97,    92,     0,    53,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   243,   239,   241,     0,
     116,     0,   133,     0,   219,   140,     0,   197,   198,     0,
       0,     0,   292,   218,     0,   214,   210,   149,   147,   157,
       0,   264,   263,   266,     0,   272,     0,   247,     0,     0,
      31,     0,    35,     0,    38,    46,    91,    59,    60,    61,
      63,    64,    66,    67,    71,    72,    69,    70,    74,    75,
      77,    79,    81,    83,    85,    87,     0,   104,   117,   118,
     135,     0,   222,     0,   139,   199,     0,   289,   293,   216,
       0,   265,     0,     0,     0,     0,     0,     0,   235,    33,
       0,   134,     0,   220,   294,   290,     0,     0,   267,     0,
     246,   244,     0,   249,     0,   237,   260,   236,    89,   221,
     223,   291,   283,     0,   268,   262,     0,     0,     0,   250,
     254,     0,   258,     0,   248,   261,   245,     0,   253,   256,
     255,   257,   251,   259,   252
};

  /* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -278,  -278,  -278,  -278,  -278,  -278,    14,   299,  -278,     9,
    -278,  -278,  -278,  -278,  -278,  -278,  -278,  -278,  -278,  -278,
     169,  -278,   -66,   -46,  -138,   -47,    33,    35,    36,    38,
      31,    37,  -278,  -150,  -205,  -278,  -121,  -166,    11,    12,
    -278,  -278,  -278,  -278,   246,   127,  -278,  -278,  -278,  -278,
     -87,     1,  -278,    93,  -278,  -278,  -278,  -278,  1889,   105,
    -278,    -9,  -128,   -13,  -278,  -278,   117,  -278,   208,  -154,
      23,    20,  -252,  -278,    96,  -153,  -277,  -278,  -278,   -52,
     320,    88,   101,  -278,  -278,    24,  -278,  -278,   -63,  -278,
     -64,  -278,  -278,  -278,  -278,  -278,  -278,   334,  -278,   -43,
    -278,   323,  -278,    42,  -278,   328,   330,   331,   332,  -278
};

  /* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     2,    13,     3,    55,     6,   265,    14,    56,   200,
     201,   202,   373,   203,   204,   205,   206,   207,   208,   209,
     210,   211,   212,   213,   214,   215,   216,   217,   218,   219,
     220,   221,   222,   223,   224,   313,   225,   260,   226,   227,
      59,    60,    61,   242,   125,   126,   127,   243,    62,    63,
      64,    99,   169,   170,   171,    66,   173,    67,    68,    69,
      70,   102,   156,   261,    73,    74,    75,    76,   161,   162,
     266,   267,   345,   403,   229,   230,   231,   232,   291,   436,
     437,   233,   234,   235,   431,   369,   236,   433,   450,   451,
     452,   453,   237,   363,   412,   413,   238,    77,    78,    79,
      80,    81,   426,   351,   352,    82,    83,    84,    85,    86
};

  /* YYTABLE[YYPACT[STATE-NUM]] -- What to do in state STATE-NUM.  If
     positive, shift that token.  If negative, reduce the rule whose
     number is the opposite.  If YYTABLE_NINF, syntax error.  */
static const yytype_int16 yytable[] =
{
      72,   160,     4,   300,    71,   259,   251,   269,   333,   447,
     448,  -287,   447,   448,    65,     7,     8,     9,    11,  -284,
    -285,  -286,   141,   279,    57,    58,    16,   155,   154,   155,
     273,   165,     1,   274,     5,     7,     8,     9,   130,    15,
       7,     8,     9,    72,   250,   344,   341,    71,    19,    98,
     103,    89,   128,    19,   131,   148,    71,    65,    87,   284,
     262,   422,    71,   322,   323,    12,    65,    57,    58,  -287,
     335,   289,   152,   135,   160,    72,   160,   346,   134,   423,
     338,    37,    38,    10,   420,    39,    37,    38,   262,   400,
      39,   402,   164,   375,   404,   166,   349,   106,   335,   275,
     334,   159,   276,   228,  -284,   401,   357,    71,   376,   141,
     128,   128,   259,   245,   336,   128,   269,   152,   128,   128,
     128,   342,   259,   354,   253,   168,   172,   244,   167,  -285,
     397,    88,   254,   255,   256,    51,   344,   353,   344,   355,
     435,   344,   335,    71,   249,   101,   101,  -286,    72,   421,
      72,   449,   435,   152,   462,   324,   325,   354,   365,   107,
     364,   108,   105,   228,   366,   367,   297,    71,   111,   298,
     440,   101,   101,   424,   374,   101,   350,   152,   129,   109,
     101,   335,   454,   160,   384,   385,   386,   387,   292,   293,
     110,   370,   112,   101,   335,   464,   344,   414,   119,   368,
     335,   101,   101,   118,   101,   415,   101,   101,   335,   115,
     116,   443,   399,   396,   335,   438,   155,   344,   338,   158,
     228,     7,     8,     9,    71,   318,   319,   294,   163,   344,
     -21,   339,   326,   327,   152,   -22,   129,   129,    42,    43,
      44,   129,   367,   252,   129,   129,   129,   240,   320,   321,
      72,   241,   380,   381,   246,   247,   248,    72,   262,   340,
     263,   430,   132,     8,   133,   350,   101,   228,   101,   264,
     -49,    71,   295,   228,   382,   383,   368,   271,   228,   388,
     389,   152,    71,   315,   316,   317,   272,   280,   168,   427,
     359,   277,   152,   456,   281,   278,   282,   441,   459,   461,
     285,   286,   290,   296,   301,   328,   330,   444,   329,   331,
     461,   -48,   332,   120,   302,   303,   304,   305,   306,   307,
     308,   309,   310,   311,   258,   347,   457,   115,   121,   122,
     123,   360,   372,   405,   -43,   407,   410,   425,   335,    72,
     417,    31,    32,    33,    34,    35,   419,   428,   429,   432,
     228,   442,   446,   398,   434,   113,   458,   455,   228,   287,
     288,   390,    71,   394,   391,   239,   392,   358,   101,   393,
     395,   270,   152,   406,   409,   101,   361,   445,   117,   371,
     314,   362,   416,   124,    42,    43,    44,   411,   460,   463,
     114,   101,   150,   142,   408,   143,   144,   145,     0,     0,
       0,   228,     0,     0,   228,    71,   312,     0,    71,     0,
       0,     0,     0,     0,     0,   152,   228,     0,   152,     0,
      71,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     152,   258,     0,   228,     0,     0,     0,    71,   228,   228,
       0,   258,    71,    71,     0,     0,  -111,   152,     0,     0,
     228,     0,   152,   152,    71,     0,     0,   101,     0,     0,
       0,     0,     0,     0,   152,     0,     0,     0,   101,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   377,   378,   379,   258,   258,   258,
     258,   258,   258,   258,   258,   258,   258,   258,   258,   258,
     258,   258,   258,    17,    18,    19,   174,    20,   175,   176,
       0,   177,   178,   179,   180,   181,     0,     0,    21,    22,
      23,    24,    25,    26,    27,    28,    29,    30,     0,     0,
       0,     0,    31,    32,    33,    34,    35,    36,    37,    38,
     182,    94,    39,    95,   183,   184,   185,   186,   187,   188,
     189,     0,     0,     0,   190,   191,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,   120,     0,     0,
       0,     0,     0,    40,    41,    42,    43,    44,     0,    45,
       0,    19,   121,   122,   123,     0,     0,     0,     0,     0,
       0,     0,    51,     0,     0,    31,    32,    33,    34,    35,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    37,    38,     0,    94,    39,    95,
     183,   184,   185,   186,   187,   188,   189,    52,    53,     0,
     190,   191,     0,     0,     0,     0,   192,   124,    42,    43,
      44,   193,   194,   195,   196,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   197,   198,   199,    17,
      18,    19,   174,    20,   175,   176,     0,   177,   178,   179,
     180,   181,     0,     0,    21,    22,    23,    24,    25,    26,
      27,    28,    29,    30,     0,     0,     0,     0,    31,    32,
      33,    34,    35,    36,    37,    38,   182,    94,    39,    95,
     183,   184,   185,   186,   187,   188,   189,     0,     0,     0,
     190,   191,   192,     0,     0,   257,     0,   193,   194,   195,
     196,     0,     0,     0,     0,     0,     0,     0,     0,    40,
      41,    42,    43,    44,     0,    45,     0,    19,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    51,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      37,    38,     0,    94,    39,    95,   183,   184,   185,   186,
     187,   188,   189,    52,    53,     0,   190,   191,     0,     0,
       0,     0,   192,     0,     0,     0,     0,   193,   194,   195,
     196,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   197,   198,   337,    17,    18,    19,   174,    20,
     175,   176,     0,   177,   178,   179,   180,   181,     0,     0,
      21,    22,    23,    24,    25,    26,    27,    28,    29,    30,
       0,     0,     0,     0,    31,    32,    33,    34,    35,    36,
      37,    38,   182,    94,    39,    95,   183,   184,   185,   186,
     187,   188,   189,     0,     0,     0,   190,   191,   192,     0,
       0,   348,     0,   193,   194,   195,   196,     0,     0,     0,
       0,     0,     0,     0,     0,    40,    41,    42,    43,    44,
       0,    45,    17,    18,    19,     0,    90,     0,     0,     0,
       0,     0,     0,     0,    51,     0,     0,    21,    91,    92,
      24,    93,    26,    27,    28,    29,    30,     0,     0,     0,
       0,    31,    32,    33,    34,    35,    36,    37,    38,     0,
       0,    39,     0,     0,     0,     0,     0,     0,     0,    52,
      53,     0,     0,     0,     0,     0,     0,     0,   192,     0,
       0,     0,     0,   193,   194,   195,   196,     0,     0,     0,
       0,     0,    96,    97,    42,    43,    44,     0,   197,   198,
     418,    17,    18,    19,     0,    90,     0,     0,     0,     0,
       0,    51,     0,     0,     0,    19,    21,    91,    92,    24,
      93,    26,    27,    28,    29,    30,     0,     0,     0,     0,
      31,    32,    33,    34,    35,    36,    37,    38,     0,     0,
      39,     0,     0,     0,     0,     0,    52,    53,    37,    38,
       0,    94,    39,    95,   183,   184,   185,   186,   187,   188,
     189,     0,     0,     0,   190,   191,     0,     0,     0,     0,
       0,    96,    97,    42,    43,    44,     0,   268,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      51,     0,     0,     0,     0,     0,     0,     0,    17,    18,
      19,   174,    20,   175,   176,     0,   177,   178,   179,   180,
     181,   447,   448,    21,    22,    23,    24,    25,    26,    27,
      28,    29,    30,     0,     0,    52,    53,    31,    32,    33,
      34,    35,    36,    37,    38,   182,    94,    39,    95,   183,
     184,   185,   186,   187,   188,   189,   192,     0,     0,   190,
     191,   193,   194,   195,   196,     0,   356,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,   343,    40,    41,
      42,    43,    44,     0,    45,    19,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    51,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    37,    38,
       0,    94,    39,    95,   183,   184,   185,   186,   187,   188,
     189,     0,    52,    53,   190,   191,     0,     0,     0,     0,
       0,   192,     0,     0,     0,     0,   193,   194,   195,   196,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   197,   198,    17,    18,    19,   174,    20,   175,   176,
       0,   177,   178,   179,   180,   181,     0,     0,    21,    22,
      23,    24,    25,    26,    27,    28,    29,    30,     0,     0,
       0,     0,    31,    32,    33,    34,    35,    36,    37,    38,
     182,    94,    39,    95,   183,   184,   185,   186,   187,   188,
     189,     0,     0,     0,   190,   191,   192,     0,     0,     0,
       0,   193,   194,   195,   196,     0,     0,     0,     0,     0,
       0,     0,     0,    40,    41,    42,    43,    44,     0,    45,
      19,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    51,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    37,   299,     0,    94,    39,    95,   183,
     184,   185,   186,   187,   188,   189,     0,    52,    53,   190,
     191,     0,     0,     0,     0,     0,   192,     0,     0,     0,
       0,   193,   194,   195,   196,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   197,   198,    17,    18,
      19,   174,    20,   175,   176,     0,   177,   178,   179,   180,
     181,     0,     0,    21,    22,    23,    24,    25,    26,    27,
      28,    29,    30,     0,     0,     0,     0,    31,    32,    33,
      34,    35,    36,    37,    38,   182,    94,    39,    95,   183,
     184,   185,   186,   187,   188,   189,     0,     0,     0,   190,
     191,   192,     0,     0,     0,    19,   193,   194,   195,   196,
       0,     0,     0,     0,     0,     0,     0,     0,    40,    41,
      42,    43,    44,     0,    45,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    51,    37,    38,
       0,    94,    39,    95,   183,   184,   185,   186,   187,   188,
     189,     0,     0,     0,   190,   191,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    52,    53,     0,    17,    18,    19,     0,    20,
       0,   192,     0,     0,     0,     0,   193,   194,   195,   196,
      21,    22,    23,    24,    25,    26,    27,    28,    29,    30,
       0,   197,   116,     0,    31,    32,    33,    34,    35,    36,
      37,    38,     0,    94,    39,    95,   183,   184,   185,   186,
     187,   188,   189,     0,     0,     0,   190,   191,     0,     0,
       0,     0,     0,     0,     0,     0,   192,     0,     0,     0,
       0,   193,   194,   195,   196,    40,    41,    42,    43,    44,
       0,    45,     0,    17,    18,    19,   283,    20,     0,     0,
       0,     0,     0,     0,    51,     0,     0,     0,    21,    22,
      23,    24,    25,    26,    27,    28,    29,    30,     0,     0,
       0,     0,    31,    32,    33,    34,    35,    36,    37,    38,
       0,     0,    39,     0,     0,     0,     0,     0,     0,    52,
      53,     0,     0,     0,     0,     0,     0,     0,   192,     0,
       0,     0,     0,   193,   194,   195,   196,     0,     0,     0,
       0,     0,     0,    40,    41,    42,    43,    44,   197,    45,
       0,    12,     0,     0,     0,     0,     0,    46,    47,    48,
      49,    50,    51,     0,     0,     0,     0,     0,     0,     0,
      17,    18,    19,     0,    20,     0,     0,     0,    19,     0,
       0,     0,     0,     0,     0,    21,    22,    23,    24,    25,
      26,    27,    28,    29,    30,     0,     0,    52,    53,    31,
      32,    33,    34,    35,    36,    37,    38,     0,     0,    39,
       0,    37,    38,     0,    94,    39,    95,   183,   184,   185,
     186,   187,   188,   189,     0,     0,    54,   190,   191,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      40,    41,    42,    43,    44,     0,    45,     0,     0,     0,
       0,     0,     0,     0,    46,    47,    48,    49,    50,    51,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    52,    53,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   192,
       0,     0,     0,     0,   193,   194,   195,   196,    17,    18,
      19,     0,    90,    54,     0,     0,     0,     0,     0,     0,
     343,   439,     0,    21,    91,    92,    24,    93,    26,    27,
      28,    29,    30,     0,     0,     0,     0,    31,    32,    33,
      34,    35,    36,    37,    38,     0,    94,    39,    95,   183,
     184,   185,   186,   187,   188,   189,     0,     0,     0,   190,
     191,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    96,    97,
      42,    43,    44,    17,    18,    19,     0,    90,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    51,    21,    91,
      92,    24,    93,    26,    27,    28,    29,    30,     0,   100,
     104,     0,    31,    32,    33,    34,    35,    36,    37,    38,
       0,     0,    39,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    52,    53,   140,   146,   147,     0,   149,   151,
     153,   192,     0,     0,   157,     0,   193,   194,   195,   196,
       0,     0,     0,    96,    97,    42,    43,    44,    17,    18,
       0,     0,    90,     0,     0,   100,   104,     0,   140,     0,
     149,   153,    51,    21,    91,    92,    24,    93,    26,    27,
      28,    29,    30,     0,     0,     0,     0,    31,    32,    33,
      34,    35,    36,     0,     0,     0,    94,     0,    95,     0,
       0,     0,     0,     0,     0,     0,     0,    52,    53,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    17,
      18,   140,     0,   136,     0,     0,     0,     0,    96,    97,
      42,    43,    44,     0,    21,   137,   138,    24,   139,    26,
      27,    28,    29,    30,     0,     0,     0,    51,    31,    32,
      33,    34,    35,    36,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      17,    18,    52,    53,    90,     0,     0,     0,     0,    96,
      97,    42,    43,    44,     0,    21,    91,    92,    24,    93,
      26,    27,    28,    29,    30,     0,     0,     0,    51,    31,
      32,    33,    34,    35,    36,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    17,    18,    52,    53,    20,     0,     0,     0,     0,
      96,    97,    42,    43,    44,     0,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,     0,     0,     0,    51,
      31,    32,    33,    34,    35,    36,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    52,    53,     0,     0,     0,     0,
       0,    96,    97,    42,    43,    44,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      51,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    52,    53
};

static const yytype_int16 yycheck[] =
{
      13,    88,    46,   208,    13,   155,   134,   161,    61,    16,
      17,    43,    16,    17,    13,    41,    42,    43,     4,    43,
      43,    43,    65,   176,    13,    13,    12,   138,    71,   138,
     137,    37,    80,   140,     0,    41,    42,    43,   140,    84,
      41,    42,    43,    56,   155,   250,   155,    56,     5,    40,
      41,    37,    61,     5,   156,    68,    65,    56,    83,   180,
     138,   140,    71,    56,    57,    81,    65,    56,    56,    43,
     140,   192,    71,    64,   161,    88,   163,   155,    64,   158,
     233,    38,    39,    84,   154,    42,    38,    39,   138,   341,
      42,   343,   105,   298,   346,   101,   262,    84,   140,   137,
     153,    87,   140,   116,    43,   155,   272,   116,   313,   152,
     119,   120,   262,   126,   156,   124,   270,   116,   127,   128,
     129,   249,   272,   140,   156,   111,   112,   126,   134,    43,
     335,   157,   156,   156,   156,    92,   341,   265,   343,   156,
     417,   346,   140,   152,   130,    40,    41,    43,   161,   401,
     163,   158,   429,   152,   158,   148,   149,   140,   156,    84,
     281,    84,    45,   176,   285,   286,   137,   176,   136,   140,
     422,    66,    67,   156,   295,    70,   263,   176,    61,    84,
      75,   140,   434,   270,   322,   323,   324,   325,    54,    55,
      84,   137,   136,    88,   140,   154,   401,   137,   140,   286,
     140,    96,    97,   137,    99,   137,   101,   102,   140,   156,
     157,   137,   340,   334,   140,   420,   138,   422,   371,    43,
     233,    41,    42,    43,   233,   141,   142,    93,   157,   434,
     136,   244,    58,    59,   233,   136,   119,   120,    75,    76,
      77,   124,   363,   136,   127,   128,   129,   120,    52,    53,
     263,   124,   318,   319,   127,   128,   129,   270,   138,   245,
     157,   414,    41,    42,    43,   352,   161,   280,   163,    84,
     136,   280,   138,   286,   320,   321,   363,   156,   291,   326,
     327,   280,   291,   145,   146,   147,   155,   136,   274,   410,
     276,   156,   291,   446,   136,   156,   156,   425,   451,   452,
     136,   136,   158,   137,   136,   150,   152,   428,   151,    60,
     463,   136,    62,     4,    63,    64,    65,    66,    67,    68,
      69,    70,    71,    72,   155,   139,   447,   156,    19,    20,
      21,    40,    51,   139,   137,   158,   136,    43,   140,   352,
     137,    32,    33,    34,    35,    36,   139,   156,   137,   157,
     363,   156,    10,   339,   155,    56,   154,   156,   371,   190,
     191,   328,   371,   332,   329,   119,   330,   274,   263,   331,
     333,   163,   371,   350,   354,   270,   280,   429,    58,   291,
     211,   280,   368,    74,    75,    76,    77,   363,   451,   453,
      56,   286,    69,    65,   352,    65,    65,    65,    -1,    -1,
      -1,   414,    -1,    -1,   417,   414,   155,    -1,   417,    -1,
      -1,    -1,    -1,    -1,    -1,   414,   429,    -1,   417,    -1,
     429,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
     429,   262,    -1,   446,    -1,    -1,    -1,   446,   451,   452,
      -1,   272,   451,   452,    -1,    -1,   137,   446,    -1,    -1,
     463,    -1,   451,   452,   463,    -1,    -1,   352,    -1,    -1,
      -1,    -1,    -1,    -1,   463,    -1,    -1,    -1,   363,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,   315,   316,   317,   318,   319,   320,
     321,   322,   323,   324,   325,   326,   327,   328,   329,   330,
     331,   332,   333,     3,     4,     5,     6,     7,     8,     9,
      -1,    11,    12,    13,    14,    15,    -1,    -1,    18,    19,
      20,    21,    22,    23,    24,    25,    26,    27,    -1,    -1,
      -1,    -1,    32,    33,    34,    35,    36,    37,    38,    39,
      40,    41,    42,    43,    44,    45,    46,    47,    48,    49,
      50,    -1,    -1,    -1,    54,    55,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,     4,    -1,    -1,
      -1,    -1,    -1,    73,    74,    75,    76,    77,    -1,    79,
      -1,     5,    19,    20,    21,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    92,    -1,    -1,    32,    33,    34,    35,    36,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    38,    39,    -1,    41,    42,    43,
      44,    45,    46,    47,    48,    49,    50,   127,   128,    -1,
      54,    55,    -1,    -1,    -1,    -1,   136,    74,    75,    76,
      77,   141,   142,   143,   144,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,   156,   157,   158,     3,
       4,     5,     6,     7,     8,     9,    -1,    11,    12,    13,
      14,    15,    -1,    -1,    18,    19,    20,    21,    22,    23,
      24,    25,    26,    27,    -1,    -1,    -1,    -1,    32,    33,
      34,    35,    36,    37,    38,    39,    40,    41,    42,    43,
      44,    45,    46,    47,    48,    49,    50,    -1,    -1,    -1,
      54,    55,   136,    -1,    -1,   139,    -1,   141,   142,   143,
     144,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    73,
      74,    75,    76,    77,    -1,    79,    -1,     5,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    92,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      38,    39,    -1,    41,    42,    43,    44,    45,    46,    47,
      48,    49,    50,   127,   128,    -1,    54,    55,    -1,    -1,
      -1,    -1,   136,    -1,    -1,    -1,    -1,   141,   142,   143,
     144,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,   156,   157,   158,     3,     4,     5,     6,     7,
       8,     9,    -1,    11,    12,    13,    14,    15,    -1,    -1,
      18,    19,    20,    21,    22,    23,    24,    25,    26,    27,
      -1,    -1,    -1,    -1,    32,    33,    34,    35,    36,    37,
      38,    39,    40,    41,    42,    43,    44,    45,    46,    47,
      48,    49,    50,    -1,    -1,    -1,    54,    55,   136,    -1,
      -1,   139,    -1,   141,   142,   143,   144,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    73,    74,    75,    76,    77,
      -1,    79,     3,     4,     5,    -1,     7,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    92,    -1,    -1,    18,    19,    20,
      21,    22,    23,    24,    25,    26,    27,    -1,    -1,    -1,
      -1,    32,    33,    34,    35,    36,    37,    38,    39,    -1,
      -1,    42,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   127,
     128,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   136,    -1,
      -1,    -1,    -1,   141,   142,   143,   144,    -1,    -1,    -1,
      -1,    -1,    73,    74,    75,    76,    77,    -1,   156,   157,
     158,     3,     4,     5,    -1,     7,    -1,    -1,    -1,    -1,
      -1,    92,    -1,    -1,    -1,     5,    18,    19,    20,    21,
      22,    23,    24,    25,    26,    27,    -1,    -1,    -1,    -1,
      32,    33,    34,    35,    36,    37,    38,    39,    -1,    -1,
      42,    -1,    -1,    -1,    -1,    -1,   127,   128,    38,    39,
      -1,    41,    42,    43,    44,    45,    46,    47,    48,    49,
      50,    -1,    -1,    -1,    54,    55,    -1,    -1,    -1,    -1,
      -1,    73,    74,    75,    76,    77,    -1,   158,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      92,    -1,    -1,    -1,    -1,    -1,    -1,    -1,     3,     4,
       5,     6,     7,     8,     9,    -1,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    -1,    -1,   127,   128,    32,    33,    34,
      35,    36,    37,    38,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    50,   136,    -1,    -1,    54,
      55,   141,   142,   143,   144,    -1,   158,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,   157,    73,    74,
      75,    76,    77,    -1,    79,     5,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    92,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    38,    39,
      -1,    41,    42,    43,    44,    45,    46,    47,    48,    49,
      50,    -1,   127,   128,    54,    55,    -1,    -1,    -1,    -1,
      -1,   136,    -1,    -1,    -1,    -1,   141,   142,   143,   144,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,   156,   157,     3,     4,     5,     6,     7,     8,     9,
      -1,    11,    12,    13,    14,    15,    -1,    -1,    18,    19,
      20,    21,    22,    23,    24,    25,    26,    27,    -1,    -1,
      -1,    -1,    32,    33,    34,    35,    36,    37,    38,    39,
      40,    41,    42,    43,    44,    45,    46,    47,    48,    49,
      50,    -1,    -1,    -1,    54,    55,   136,    -1,    -1,    -1,
      -1,   141,   142,   143,   144,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    73,    74,    75,    76,    77,    -1,    79,
       5,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    92,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    38,    39,    -1,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    50,    -1,   127,   128,    54,
      55,    -1,    -1,    -1,    -1,    -1,   136,    -1,    -1,    -1,
      -1,   141,   142,   143,   144,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,   156,   157,     3,     4,
       5,     6,     7,     8,     9,    -1,    11,    12,    13,    14,
      15,    -1,    -1,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    -1,    -1,    -1,    -1,    32,    33,    34,
      35,    36,    37,    38,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    50,    -1,    -1,    -1,    54,
      55,   136,    -1,    -1,    -1,     5,   141,   142,   143,   144,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    73,    74,
      75,    76,    77,    -1,    79,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    92,    38,    39,
      -1,    41,    42,    43,    44,    45,    46,    47,    48,    49,
      50,    -1,    -1,    -1,    54,    55,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,   127,   128,    -1,     3,     4,     5,    -1,     7,
      -1,   136,    -1,    -1,    -1,    -1,   141,   142,   143,   144,
      18,    19,    20,    21,    22,    23,    24,    25,    26,    27,
      -1,   156,   157,    -1,    32,    33,    34,    35,    36,    37,
      38,    39,    -1,    41,    42,    43,    44,    45,    46,    47,
      48,    49,    50,    -1,    -1,    -1,    54,    55,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,   136,    -1,    -1,    -1,
      -1,   141,   142,   143,   144,    73,    74,    75,    76,    77,
      -1,    79,    -1,     3,     4,     5,   156,     7,    -1,    -1,
      -1,    -1,    -1,    -1,    92,    -1,    -1,    -1,    18,    19,
      20,    21,    22,    23,    24,    25,    26,    27,    -1,    -1,
      -1,    -1,    32,    33,    34,    35,    36,    37,    38,    39,
      -1,    -1,    42,    -1,    -1,    -1,    -1,    -1,    -1,   127,
     128,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   136,    -1,
      -1,    -1,    -1,   141,   142,   143,   144,    -1,    -1,    -1,
      -1,    -1,    -1,    73,    74,    75,    76,    77,   156,    79,
      -1,    81,    -1,    -1,    -1,    -1,    -1,    87,    88,    89,
      90,    91,    92,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
       3,     4,     5,    -1,     7,    -1,    -1,    -1,     5,    -1,
      -1,    -1,    -1,    -1,    -1,    18,    19,    20,    21,    22,
      23,    24,    25,    26,    27,    -1,    -1,   127,   128,    32,
      33,    34,    35,    36,    37,    38,    39,    -1,    -1,    42,
      -1,    38,    39,    -1,    41,    42,    43,    44,    45,    46,
      47,    48,    49,    50,    -1,    -1,   156,    54,    55,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      73,    74,    75,    76,    77,    -1,    79,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    87,    88,    89,    90,    91,    92,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,   127,   128,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   136,
      -1,    -1,    -1,    -1,   141,   142,   143,   144,     3,     4,
       5,    -1,     7,   156,    -1,    -1,    -1,    -1,    -1,    -1,
     157,   158,    -1,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    -1,    -1,    -1,    -1,    32,    33,    34,
      35,    36,    37,    38,    39,    -1,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    50,    -1,    -1,    -1,    54,
      55,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    73,    74,
      75,    76,    77,     3,     4,     5,    -1,     7,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    92,    18,    19,
      20,    21,    22,    23,    24,    25,    26,    27,    -1,    40,
      41,    -1,    32,    33,    34,    35,    36,    37,    38,    39,
      -1,    -1,    42,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,   127,   128,    65,    66,    67,    -1,    69,    70,
      71,   136,    -1,    -1,    75,    -1,   141,   142,   143,   144,
      -1,    -1,    -1,    73,    74,    75,    76,    77,     3,     4,
      -1,    -1,     7,    -1,    -1,    96,    97,    -1,    99,    -1,
     101,   102,    92,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    -1,    -1,    -1,    -1,    32,    33,    34,
      35,    36,    37,    -1,    -1,    -1,    41,    -1,    43,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,   127,   128,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,     3,
       4,   152,    -1,     7,    -1,    -1,    -1,    -1,    73,    74,
      75,    76,    77,    -1,    18,    19,    20,    21,    22,    23,
      24,    25,    26,    27,    -1,    -1,    -1,    92,    32,    33,
      34,    35,    36,    37,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
       3,     4,   127,   128,     7,    -1,    -1,    -1,    -1,    73,
      74,    75,    76,    77,    -1,    18,    19,    20,    21,    22,
      23,    24,    25,    26,    27,    -1,    -1,    -1,    92,    32,
      33,    34,    35,    36,    37,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,     3,     4,   127,   128,     7,    -1,    -1,    -1,    -1,
      73,    74,    75,    76,    77,    -1,    18,    19,    20,    21,
      22,    23,    24,    25,    26,    27,    -1,    -1,    -1,    92,
      32,    33,    34,    35,    36,    37,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,   127,   128,    -1,    -1,    -1,    -1,
      -1,    73,    74,    75,    76,    77,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      92,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,   127,   128
};

  /* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
     symbol of state STATE-NUM.  */
static const yytype_uint16 yystos[] =
{
       0,    80,   160,   162,    46,     0,   164,    41,    42,    43,
      84,   165,    81,   161,   166,    84,   165,     3,     4,     5,
       7,    18,    19,    20,    21,    22,    23,    24,    25,    26,
      27,    32,    33,    34,    35,    36,    37,    38,    39,    42,
      73,    74,    75,    76,    77,    79,    87,    88,    89,    90,
      91,    92,   127,   128,   156,   163,   167,   197,   198,   199,
     200,   201,   207,   208,   209,   210,   214,   216,   217,   218,
     219,   220,   222,   223,   224,   225,   226,   256,   257,   258,
     259,   260,   264,   265,   266,   267,   268,    83,   157,   165,
       7,    19,    20,    22,    41,    43,    73,    74,   168,   210,
     217,   218,   220,   168,   217,   225,    84,    84,    84,    84,
      84,   136,   136,   166,   256,   156,   157,   239,   137,   140,
       4,    19,    20,    21,    74,   203,   204,   205,   220,   225,
     140,   156,    41,    43,   165,   168,     7,    19,    20,    22,
     217,   258,   264,   265,   266,   267,   217,   217,   222,   217,
     260,   217,   210,   217,   258,   138,   221,   217,    43,   165,
     209,   227,   228,   157,   222,    37,   101,   134,   165,   211,
     212,   213,   165,   215,     6,     8,     9,    11,    12,    13,
      14,    15,    40,    44,    45,    46,    47,    48,    49,    50,
      54,    55,   136,   141,   142,   143,   144,   156,   157,   158,
     168,   169,   170,   172,   173,   174,   175,   176,   177,   178,
     179,   180,   181,   182,   183,   184,   185,   186,   187,   188,
     189,   190,   191,   192,   193,   195,   197,   198,   222,   233,
     234,   235,   236,   240,   241,   242,   245,   251,   255,   203,
     204,   204,   202,   206,   210,   222,   204,   204,   204,   165,
     155,   221,   136,   156,   156,   156,   156,   139,   179,   192,
     196,   222,   138,   157,    84,   165,   229,   230,   158,   228,
     227,   156,   155,   137,   140,   137,   140,   156,   156,   234,
     136,   136,   156,   156,   195,   136,   136,   179,   179,   195,
     158,   237,    54,    55,    93,   138,   137,   137,   140,    39,
     193,   136,    63,    64,    65,    66,    67,    68,    69,    70,
      71,    72,   155,   194,   179,   145,   146,   147,   141,   142,
      52,    53,    56,    57,   148,   149,    58,    59,   150,   151,
     152,    60,    62,    61,   153,   140,   156,   158,   234,   222,
     165,   155,   221,   157,   193,   231,   155,   139,   139,   196,
     209,   262,   263,   221,   140,   156,   158,   196,   212,   165,
      40,   233,   241,   252,   195,   156,   195,   195,   209,   244,
     137,   240,    51,   171,   195,   193,   193,   179,   179,   179,
     181,   181,   182,   182,   183,   183,   183,   183,   184,   184,
     185,   186,   187,   188,   189,   190,   195,   193,   165,   221,
     231,   155,   231,   232,   231,   139,   229,   158,   262,   230,
     136,   244,   253,   254,   137,   137,   165,   137,   158,   139,
     154,   231,   140,   158,   156,    43,   261,   195,   156,   137,
     234,   243,   157,   246,   155,   235,   238,   239,   193,   158,
     231,   221,   156,   137,   195,   238,    10,    16,    17,   158,
     247,   248,   249,   250,   231,   156,   234,   195,   154,   234,
     247,   234,   158,   249,   154
};

  /* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint16 yyr1[] =
{
       0,   159,   161,   160,   162,   162,   162,   163,   163,   163,
     163,   163,   164,   164,   165,   165,   165,   166,   167,   167,
     167,   168,   168,   169,   169,   169,   169,   169,   169,   169,
     169,   169,   170,   170,   170,   170,   170,   170,   171,   172,
     173,   174,   174,   175,   175,   176,   176,   177,   178,   178,
     179,   179,   179,   179,   180,   180,   180,   180,   181,   181,
     181,   181,   182,   182,   182,   183,   183,   183,   184,   184,
     184,   184,   184,   185,   185,   185,   186,   186,   187,   187,
     188,   188,   189,   189,   190,   190,   191,   191,   192,   192,
     193,   193,   194,   194,   194,   194,   194,   194,   194,   194,
     194,   194,   194,   195,   195,   196,   197,   197,   197,   197,
     198,   199,   199,   200,   200,   201,   202,   202,   202,   203,
     203,   204,   204,   204,   204,   204,   204,   205,   205,   205,
     206,   207,   207,   207,   207,   207,   208,   208,   208,   208,
     208,   208,   208,   209,   209,   210,   211,   211,   212,   212,
     212,   213,   213,   213,   214,   214,   215,   215,   216,   216,
     216,   217,   217,   217,   217,   217,   217,   217,   217,   217,
     217,   217,   217,   217,   217,   217,   217,   217,   217,   218,
     218,   218,   219,   219,   219,   219,   219,   219,   219,   219,
     219,   220,   220,   220,   220,   220,   221,   221,   221,   221,
     222,   222,   223,   223,   223,   224,   224,   225,   225,   225,
     226,   226,   227,   227,   228,   229,   229,   230,   230,   231,
     231,   231,   232,   232,   233,   234,   234,   235,   235,   235,
     235,   235,   235,   236,   237,   236,   238,   238,   239,   239,
     240,   240,   241,   241,   242,   243,   243,   244,   244,   245,
     246,   246,   247,   247,   248,   248,   249,   249,   250,   250,
     251,   251,   251,   252,   252,   253,   253,   254,   254,   255,
     255,   255,   255,   255,   256,   256,   256,   256,   256,   257,
     258,   258,   258,   259,   260,   260,   260,   260,   260,   261,
     261,   261,   262,   262,   263,   264,   264,   265,   265,   266,
     266,   267,   267,   268,   268,   268,   268
};

  /* YYR2[YYN] -- Number of symbols on the right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     0,     4,     0,     3,     4,     2,     2,     2,
       2,     2,     0,     2,     1,     1,     1,     5,     1,     2,
       2,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     3,     1,     4,     1,     3,     2,     2,     1,     1,
       1,     2,     2,     2,     1,     2,     3,     2,     1,     1,
       1,     2,     2,     2,     1,     1,     1,     1,     1,     3,
       3,     3,     1,     3,     3,     1,     3,     3,     1,     3,
       3,     3,     3,     1,     3,     3,     1,     3,     1,     3,
       1,     3,     1,     3,     1,     3,     1,     3,     1,     5,
       1,     3,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     3,     1,     2,     2,     4,     1,
       2,     1,     1,     2,     3,     3,     2,     3,     3,     2,
       2,     0,     2,     2,     2,     2,     2,     1,     1,     1,
       1,     1,     3,     4,     6,     5,     1,     2,     3,     5,
       4,     2,     2,     1,     2,     4,     1,     3,     1,     3,
       1,     1,     1,     1,     1,     4,     1,     3,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     2,     3,     3,     4,
       1,     2,     1,     1,     1,     1,     1,     1,     1,     1,
       5,     4,     1,     2,     3,     1,     3,     1,     2,     1,
       3,     4,     1,     3,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     2,     0,     4,     1,     1,     2,     3,
       1,     2,     1,     2,     5,     3,     1,     1,     4,     5,
       2,     3,     3,     2,     1,     2,     2,     2,     1,     2,
       5,     7,     6,     1,     1,     1,     0,     2,     3,     2,
       2,     2,     3,     2,     1,     1,     1,     1,     1,     2,
       1,     2,     2,     7,     1,     1,     1,     1,     2,     0,
       1,     2,     1,     2,     3,     2,     3,     2,     3,     2,
       3,     2,     3,     1,     1,     1,     1
};


#define yyerrok         (yyerrstatus = 0)
#define yyclearin       (yychar = YYEMPTY)
#define YYEMPTY         (-2)
#define YYEOF           0

#define YYACCEPT        goto yyacceptlab
#define YYABORT         goto yyabortlab
#define YYERROR         goto yyerrorlab


#define YYRECOVERING()  (!!yyerrstatus)

#define YYBACKUP(Token, Value)                                  \
do                                                              \
  if (yychar == YYEMPTY)                                        \
    {                                                           \
      yychar = (Token);                                         \
      yylval = (Value);                                         \
      YYPOPSTACK (yylen);                                       \
      yystate = *yyssp;                                         \
      goto yybackup;                                            \
    }                                                           \
  else                                                          \
    {                                                           \
      yyerror (&yylloc, state, YY_("syntax error: cannot back up")); \
      YYERROR;                                                  \
    }                                                           \
while (0)

/* Error token number */
#define YYTERROR        1
#define YYERRCODE       256


/* YYLLOC_DEFAULT -- Set CURRENT to span from RHS[1] to RHS[N].
   If N is 0, then set CURRENT to the empty location which ends
   the previous symbol: RHS[0] (always defined).  */

#ifndef YYLLOC_DEFAULT
# define YYLLOC_DEFAULT(Current, Rhs, N)                                \
    do                                                                  \
      if (N)                                                            \
        {                                                               \
          (Current).first_line   = YYRHSLOC (Rhs, 1).first_line;        \
          (Current).first_column = YYRHSLOC (Rhs, 1).first_column;      \
          (Current).last_line    = YYRHSLOC (Rhs, N).last_line;         \
          (Current).last_column  = YYRHSLOC (Rhs, N).last_column;       \
        }                                                               \
      else                                                              \
        {                                                               \
          (Current).first_line   = (Current).last_line   =              \
            YYRHSLOC (Rhs, 0).last_line;                                \
          (Current).first_column = (Current).last_column =              \
            YYRHSLOC (Rhs, 0).last_column;                              \
        }                                                               \
    while (0)
#endif

#define YYRHSLOC(Rhs, K) ((Rhs)[K])


/* Enable debugging if requested.  */
#if YYDEBUG

# ifndef YYFPRINTF
#  include <stdio.h> /* INFRINGES ON USER NAME SPACE */
#  define YYFPRINTF fprintf
# endif

# define YYDPRINTF(Args)                        \
do {                                            \
  if (yydebug)                                  \
    YYFPRINTF Args;                             \
} while (0)


/* YY_LOCATION_PRINT -- Print the location on the stream.
   This macro was not mandated originally: define only if we know
   we won't break user code: when these are the locations we know.  */

#ifndef YY_LOCATION_PRINT
# if defined YYLTYPE_IS_TRIVIAL && YYLTYPE_IS_TRIVIAL

/* Print *YYLOCP on YYO.  Private, do not rely on its existence. */

YY_ATTRIBUTE_UNUSED
static unsigned
yy_location_print_ (FILE *yyo, YYLTYPE const * const yylocp)
{
  unsigned res = 0;
  int end_col = 0 != yylocp->last_column ? yylocp->last_column - 1 : 0;
  if (0 <= yylocp->first_line)
    {
      res += YYFPRINTF (yyo, "%d", yylocp->first_line);
      if (0 <= yylocp->first_column)
        res += YYFPRINTF (yyo, ".%d", yylocp->first_column);
    }
  if (0 <= yylocp->last_line)
    {
      if (yylocp->first_line < yylocp->last_line)
        {
          res += YYFPRINTF (yyo, "-%d", yylocp->last_line);
          if (0 <= end_col)
            res += YYFPRINTF (yyo, ".%d", end_col);
        }
      else if (0 <= end_col && yylocp->first_column < end_col)
        res += YYFPRINTF (yyo, "-%d", end_col);
    }
  return res;
 }

#  define YY_LOCATION_PRINT(File, Loc)          \
  yy_location_print_ (File, &(Loc))

# else
#  define YY_LOCATION_PRINT(File, Loc) ((void) 0)
# endif
#endif


# define YY_SYMBOL_PRINT(Title, Type, Value, Location)                    \
do {                                                                      \
  if (yydebug)                                                            \
    {                                                                     \
      YYFPRINTF (stderr, "%s ", Title);                                   \
      yy_symbol_print (stderr,                                            \
                  Type, Value, Location, state); \
      YYFPRINTF (stderr, "\n");                                           \
    }                                                                     \
} while (0)


/*----------------------------------------.
| Print this symbol's value on YYOUTPUT.  |
`----------------------------------------*/

static void
yy_symbol_value_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep, YYLTYPE const * const yylocationp, struct _mesa_glsl_parse_state *state)
{
  FILE *yyo = yyoutput;
  YYUSE (yyo);
  YYUSE (yylocationp);
  YYUSE (state);
  if (!yyvaluep)
    return;
# ifdef YYPRINT
  if (yytype < YYNTOKENS)
    YYPRINT (yyoutput, yytoknum[yytype], *yyvaluep);
# endif
  YYUSE (yytype);
}


/*--------------------------------.
| Print this symbol on YYOUTPUT.  |
`--------------------------------*/

static void
yy_symbol_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep, YYLTYPE const * const yylocationp, struct _mesa_glsl_parse_state *state)
{
  YYFPRINTF (yyoutput, "%s %s (",
             yytype < YYNTOKENS ? "token" : "nterm", yytname[yytype]);

  YY_LOCATION_PRINT (yyoutput, *yylocationp);
  YYFPRINTF (yyoutput, ": ");
  yy_symbol_value_print (yyoutput, yytype, yyvaluep, yylocationp, state);
  YYFPRINTF (yyoutput, ")");
}

/*------------------------------------------------------------------.
| yy_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (included).                                                   |
`------------------------------------------------------------------*/

static void
yy_stack_print (yytype_int16 *yybottom, yytype_int16 *yytop)
{
  YYFPRINTF (stderr, "Stack now");
  for (; yybottom <= yytop; yybottom++)
    {
      int yybot = *yybottom;
      YYFPRINTF (stderr, " %d", yybot);
    }
  YYFPRINTF (stderr, "\n");
}

# define YY_STACK_PRINT(Bottom, Top)                            \
do {                                                            \
  if (yydebug)                                                  \
    yy_stack_print ((Bottom), (Top));                           \
} while (0)


/*------------------------------------------------.
| Report that the YYRULE is going to be reduced.  |
`------------------------------------------------*/

static void
yy_reduce_print (yytype_int16 *yyssp, YYSTYPE *yyvsp, YYLTYPE *yylsp, int yyrule, struct _mesa_glsl_parse_state *state)
{
  unsigned long yylno = yyrline[yyrule];
  int yynrhs = yyr2[yyrule];
  int yyi;
  YYFPRINTF (stderr, "Reducing stack by rule %d (line %lu):\n",
             yyrule - 1, yylno);
  /* The symbols being reduced.  */
  for (yyi = 0; yyi < yynrhs; yyi++)
    {
      YYFPRINTF (stderr, "   $%d = ", yyi + 1);
      yy_symbol_print (stderr,
                       yystos[yyssp[yyi + 1 - yynrhs]],
                       &(yyvsp[(yyi + 1) - (yynrhs)])
                       , &(yylsp[(yyi + 1) - (yynrhs)])                       , state);
      YYFPRINTF (stderr, "\n");
    }
}

# define YY_REDUCE_PRINT(Rule)          \
do {                                    \
  if (yydebug)                          \
    yy_reduce_print (yyssp, yyvsp, yylsp, Rule, state); \
} while (0)

/* Nonzero means print parse trace.  It is left uninitialized so that
   multiple parsers can coexist.  */
int yydebug;
#else /* !YYDEBUG */
# define YYDPRINTF(Args)
# define YY_SYMBOL_PRINT(Title, Type, Value, Location)
# define YY_STACK_PRINT(Bottom, Top)
# define YY_REDUCE_PRINT(Rule)
#endif /* !YYDEBUG */


/* YYINITDEPTH -- initial size of the parser's stacks.  */
#ifndef YYINITDEPTH
# define YYINITDEPTH 200
#endif

/* YYMAXDEPTH -- maximum size the stacks can grow to (effective only
   if the built-in stack extension method is used).

   Do not make this value too large; the results are undefined if
   YYSTACK_ALLOC_MAXIMUM < YYSTACK_BYTES (YYMAXDEPTH)
   evaluated with infinite-precision integer arithmetic.  */

#ifndef YYMAXDEPTH
# define YYMAXDEPTH 10000
#endif


#if YYERROR_VERBOSE

# ifndef yystrlen
#  if defined __GLIBC__ && defined _STRING_H
#   define yystrlen strlen
#  else
/* Return the length of YYSTR.  */
static YYSIZE_T
yystrlen (const char *yystr)
{
  YYSIZE_T yylen;
  for (yylen = 0; yystr[yylen]; yylen++)
    continue;
  return yylen;
}
#  endif
# endif

# ifndef yystpcpy
#  if defined __GLIBC__ && defined _STRING_H && defined _GNU_SOURCE
#   define yystpcpy stpcpy
#  else
/* Copy YYSRC to YYDEST, returning the address of the terminating '\0' in
   YYDEST.  */
static char *
yystpcpy (char *yydest, const char *yysrc)
{
  char *yyd = yydest;
  const char *yys = yysrc;

  while ((*yyd++ = *yys++) != '\0')
    continue;

  return yyd - 1;
}
#  endif
# endif

# ifndef yytnamerr
/* Copy to YYRES the contents of YYSTR after stripping away unnecessary
   quotes and backslashes, so that it's suitable for yyerror.  The
   heuristic is that double-quoting is unnecessary unless the string
   contains an apostrophe, a comma, or backslash (other than
   backslash-backslash).  YYSTR is taken from yytname.  If YYRES is
   null, do not copy; instead, return the length of what the result
   would have been.  */
static YYSIZE_T
yytnamerr (char *yyres, const char *yystr)
{
  if (*yystr == '"')
    {
      YYSIZE_T yyn = 0;
      char const *yyp = yystr;

      for (;;)
        switch (*++yyp)
          {
          case '\'':
          case ',':
            goto do_not_strip_quotes;

          case '\\':
            if (*++yyp != '\\')
              goto do_not_strip_quotes;
            /* Fall through.  */
          default:
            if (yyres)
              yyres[yyn] = *yyp;
            yyn++;
            break;

          case '"':
            if (yyres)
              yyres[yyn] = '\0';
            return yyn;
          }
    do_not_strip_quotes: ;
    }

  if (! yyres)
    return yystrlen (yystr);

  return yystpcpy (yyres, yystr) - yyres;
}
# endif

/* Copy into *YYMSG, which is of size *YYMSG_ALLOC, an error message
   about the unexpected token YYTOKEN for the state stack whose top is
   YYSSP.

   Return 0 if *YYMSG was successfully written.  Return 1 if *YYMSG is
   not large enough to hold the message.  In that case, also set
   *YYMSG_ALLOC to the required number of bytes.  Return 2 if the
   required number of bytes is too large to store.  */
static int
yysyntax_error (YYSIZE_T *yymsg_alloc, char **yymsg,
                yytype_int16 *yyssp, int yytoken)
{
  YYSIZE_T yysize0 = yytnamerr (YY_NULLPTR, yytname[yytoken]);
  YYSIZE_T yysize = yysize0;
  enum { YYERROR_VERBOSE_ARGS_MAXIMUM = 5 };
  /* Internationalized format string. */
  const char *yyformat = YY_NULLPTR;
  /* Arguments of yyformat. */
  char const *yyarg[YYERROR_VERBOSE_ARGS_MAXIMUM];
  /* Number of reported tokens (one for the "unexpected", one per
     "expected"). */
  int yycount = 0;

  /* There are many possibilities here to consider:
     - If this state is a consistent state with a default action, then
       the only way this function was invoked is if the default action
       is an error action.  In that case, don't check for expected
       tokens because there are none.
     - The only way there can be no lookahead present (in yychar) is if
       this state is a consistent state with a default action.  Thus,
       detecting the absence of a lookahead is sufficient to determine
       that there is no unexpected or expected token to report.  In that
       case, just report a simple "syntax error".
     - Don't assume there isn't a lookahead just because this state is a
       consistent state with a default action.  There might have been a
       previous inconsistent state, consistent state with a non-default
       action, or user semantic action that manipulated yychar.
     - Of course, the expected token list depends on states to have
       correct lookahead information, and it depends on the parser not
       to perform extra reductions after fetching a lookahead from the
       scanner and before detecting a syntax error.  Thus, state merging
       (from LALR or IELR) and default reductions corrupt the expected
       token list.  However, the list is correct for canonical LR with
       one exception: it will still contain any token that will not be
       accepted due to an error action in a later state.
  */
  if (yytoken != YYEMPTY)
    {
      int yyn = yypact[*yyssp];
      yyarg[yycount++] = yytname[yytoken];
      if (!yypact_value_is_default (yyn))
        {
          /* Start YYX at -YYN if negative to avoid negative indexes in
             YYCHECK.  In other words, skip the first -YYN actions for
             this state because they are default actions.  */
          int yyxbegin = yyn < 0 ? -yyn : 0;
          /* Stay within bounds of both yycheck and yytname.  */
          int yychecklim = YYLAST - yyn + 1;
          int yyxend = yychecklim < YYNTOKENS ? yychecklim : YYNTOKENS;
          int yyx;

          for (yyx = yyxbegin; yyx < yyxend; ++yyx)
            if (yycheck[yyx + yyn] == yyx && yyx != YYTERROR
                && !yytable_value_is_error (yytable[yyx + yyn]))
              {
                if (yycount == YYERROR_VERBOSE_ARGS_MAXIMUM)
                  {
                    yycount = 1;
                    yysize = yysize0;
                    break;
                  }
                yyarg[yycount++] = yytname[yyx];
                {
                  YYSIZE_T yysize1 = yysize + yytnamerr (YY_NULLPTR, yytname[yyx]);
                  if (! (yysize <= yysize1
                         && yysize1 <= YYSTACK_ALLOC_MAXIMUM))
                    return 2;
                  yysize = yysize1;
                }
              }
        }
    }

  switch (yycount)
    {
# define YYCASE_(N, S)                      \
      case N:                               \
        yyformat = S;                       \
      break
    default: /* Avoid compiler warnings. */
      YYCASE_(0, YY_("syntax error"));
      YYCASE_(1, YY_("syntax error, unexpected %s"));
      YYCASE_(2, YY_("syntax error, unexpected %s, expecting %s"));
      YYCASE_(3, YY_("syntax error, unexpected %s, expecting %s or %s"));
      YYCASE_(4, YY_("syntax error, unexpected %s, expecting %s or %s or %s"));
      YYCASE_(5, YY_("syntax error, unexpected %s, expecting %s or %s or %s or %s"));
# undef YYCASE_
    }

  {
    YYSIZE_T yysize1 = yysize + yystrlen (yyformat);
    if (! (yysize <= yysize1 && yysize1 <= YYSTACK_ALLOC_MAXIMUM))
      return 2;
    yysize = yysize1;
  }

  if (*yymsg_alloc < yysize)
    {
      *yymsg_alloc = 2 * yysize;
      if (! (yysize <= *yymsg_alloc
             && *yymsg_alloc <= YYSTACK_ALLOC_MAXIMUM))
        *yymsg_alloc = YYSTACK_ALLOC_MAXIMUM;
      return 1;
    }

  /* Avoid sprintf, as that infringes on the user's name space.
     Don't have undefined behavior even if the translation
     produced a string with the wrong number of "%s"s.  */
  {
    char *yyp = *yymsg;
    int yyi = 0;
    while ((*yyp = *yyformat) != '\0')
      if (*yyp == '%' && yyformat[1] == 's' && yyi < yycount)
        {
          yyp += yytnamerr (yyp, yyarg[yyi++]);
          yyformat += 2;
        }
      else
        {
          yyp++;
          yyformat++;
        }
  }
  return 0;
}
#endif /* YYERROR_VERBOSE */

/*-----------------------------------------------.
| Release the memory associated to this symbol.  |
`-----------------------------------------------*/

static void
yydestruct (const char *yymsg, int yytype, YYSTYPE *yyvaluep, YYLTYPE *yylocationp, struct _mesa_glsl_parse_state *state)
{
  YYUSE (yyvaluep);
  YYUSE (yylocationp);
  YYUSE (state);
  if (!yymsg)
    yymsg = "Deleting";
  YY_SYMBOL_PRINT (yymsg, yytype, yyvaluep, yylocationp);

  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  YYUSE (yytype);
  YY_IGNORE_MAYBE_UNINITIALIZED_END
}




/*----------.
| yyparse.  |
`----------*/

int
yyparse (struct _mesa_glsl_parse_state *state)
{
/* The lookahead symbol.  */
int yychar;


/* The semantic value of the lookahead symbol.  */
/* Default value used for initialization, for pacifying older GCCs
   or non-GCC compilers.  */
YY_INITIAL_VALUE (static YYSTYPE yyval_default;)
YYSTYPE yylval YY_INITIAL_VALUE (= yyval_default);

/* Location data for the lookahead symbol.  */
static YYLTYPE yyloc_default
# if defined YYLTYPE_IS_TRIVIAL && YYLTYPE_IS_TRIVIAL
  = { 1, 1, 1, 1 }
# endif
;
YYLTYPE yylloc = yyloc_default;

    /* Number of syntax errors so far.  */
    int yynerrs;

    int yystate;
    /* Number of tokens to shift before error messages enabled.  */
    int yyerrstatus;

    /* The stacks and their tools:
       'yyss': related to states.
       'yyvs': related to semantic values.
       'yyls': related to locations.

       Refer to the stacks through separate pointers, to allow yyoverflow
       to reallocate them elsewhere.  */

    /* The state stack.  */
    yytype_int16 yyssa[YYINITDEPTH];
    yytype_int16 *yyss;
    yytype_int16 *yyssp;

    /* The semantic value stack.  */
    YYSTYPE yyvsa[YYINITDEPTH];
    YYSTYPE *yyvs;
    YYSTYPE *yyvsp;

    /* The location stack.  */
    YYLTYPE yylsa[YYINITDEPTH];
    YYLTYPE *yyls;
    YYLTYPE *yylsp;

    /* The locations where the error started and ended.  */
    YYLTYPE yyerror_range[3];

    YYSIZE_T yystacksize;

  int yyn;
  int yyresult;
  /* Lookahead token as an internal (translated) token number.  */
  int yytoken = 0;
  /* The variables used to return semantic value and location from the
     action routines.  */
  YYSTYPE yyval;
  YYLTYPE yyloc;

#if YYERROR_VERBOSE
  /* Buffer for error messages, and its allocated size.  */
  char yymsgbuf[128];
  char *yymsg = yymsgbuf;
  YYSIZE_T yymsg_alloc = sizeof yymsgbuf;
#endif

#define YYPOPSTACK(N)   (yyvsp -= (N), yyssp -= (N), yylsp -= (N))

  /* The number of symbols on the RHS of the reduced rule.
     Keep to zero when no symbol should be popped.  */
  int yylen = 0;

  yyssp = yyss = yyssa;
  yyvsp = yyvs = yyvsa;
  yylsp = yyls = yylsa;
  yystacksize = YYINITDEPTH;

  YYDPRINTF ((stderr, "Starting parse\n"));

  yystate = 0;
  yyerrstatus = 0;
  yynerrs = 0;
  yychar = YYEMPTY; /* Cause a token to be read.  */

/* User initialization code.  */
#line 87 "./glsl/glsl_parser.yy" /* yacc.c:1433  */
{
   yylloc.first_line = 1;
   yylloc.first_column = 1;
   yylloc.last_line = 1;
   yylloc.last_column = 1;
   yylloc.source = 0;
}

#line 2190 "glsl/glsl_parser.cpp" /* yacc.c:1433  */
  yylsp[0] = yylloc;
  goto yysetstate;

/*------------------------------------------------------------.
| yynewstate -- Push a new state, which is found in yystate.  |
`------------------------------------------------------------*/
 yynewstate:
  /* In all cases, when you get here, the value and location stacks
     have just been pushed.  So pushing a state here evens the stacks.  */
  yyssp++;

 yysetstate:
  *yyssp = yystate;

  if (yyss + yystacksize - 1 <= yyssp)
    {
      /* Get the current used size of the three stacks, in elements.  */
      YYSIZE_T yysize = yyssp - yyss + 1;

#ifdef yyoverflow
      {
        /* Give user a chance to reallocate the stack.  Use copies of
           these so that the &'s don't force the real ones into
           memory.  */
        YYSTYPE *yyvs1 = yyvs;
        yytype_int16 *yyss1 = yyss;
        YYLTYPE *yyls1 = yyls;

        /* Each stack pointer address is followed by the size of the
           data in use in that stack, in bytes.  This used to be a
           conditional around just the two extra args, but that might
           be undefined if yyoverflow is a macro.  */
        yyoverflow (YY_("memory exhausted"),
                    &yyss1, yysize * sizeof (*yyssp),
                    &yyvs1, yysize * sizeof (*yyvsp),
                    &yyls1, yysize * sizeof (*yylsp),
                    &yystacksize);

        yyls = yyls1;
        yyss = yyss1;
        yyvs = yyvs1;
      }
#else /* no yyoverflow */
# ifndef YYSTACK_RELOCATE
      goto yyexhaustedlab;
# else
      /* Extend the stack our own way.  */
      if (YYMAXDEPTH <= yystacksize)
        goto yyexhaustedlab;
      yystacksize *= 2;
      if (YYMAXDEPTH < yystacksize)
        yystacksize = YYMAXDEPTH;

      {
        yytype_int16 *yyss1 = yyss;
        union yyalloc *yyptr =
          (union yyalloc *) YYSTACK_ALLOC (YYSTACK_BYTES (yystacksize));
        if (! yyptr)
          goto yyexhaustedlab;
        YYSTACK_RELOCATE (yyss_alloc, yyss);
        YYSTACK_RELOCATE (yyvs_alloc, yyvs);
        YYSTACK_RELOCATE (yyls_alloc, yyls);
#  undef YYSTACK_RELOCATE
        if (yyss1 != yyssa)
          YYSTACK_FREE (yyss1);
      }
# endif
#endif /* no yyoverflow */

      yyssp = yyss + yysize - 1;
      yyvsp = yyvs + yysize - 1;
      yylsp = yyls + yysize - 1;

      YYDPRINTF ((stderr, "Stack size increased to %lu\n",
                  (unsigned long) yystacksize));

      if (yyss + yystacksize - 1 <= yyssp)
        YYABORT;
    }

  YYDPRINTF ((stderr, "Entering state %d\n", yystate));

  if (yystate == YYFINAL)
    YYACCEPT;

  goto yybackup;

/*-----------.
| yybackup.  |
`-----------*/
yybackup:

  /* Do appropriate processing given the current state.  Read a
     lookahead token if we need one and don't already have one.  */

  /* First try to decide what to do without reference to lookahead token.  */
  yyn = yypact[yystate];
  if (yypact_value_is_default (yyn))
    goto yydefault;

  /* Not known => get a lookahead token if don't already have one.  */

  /* YYCHAR is either YYEMPTY or YYEOF or a valid lookahead symbol.  */
  if (yychar == YYEMPTY)
    {
      YYDPRINTF ((stderr, "Reading a token: "));
      yychar = yylex (&yylval, &yylloc, state);
    }

  if (yychar <= YYEOF)
    {
      yychar = yytoken = YYEOF;
      YYDPRINTF ((stderr, "Now at end of input.\n"));
    }
  else
    {
      yytoken = YYTRANSLATE (yychar);
      YY_SYMBOL_PRINT ("Next token is", yytoken, &yylval, &yylloc);
    }

  /* If the proper action on seeing token YYTOKEN is to reduce or to
     detect an error, take that action.  */
  yyn += yytoken;
  if (yyn < 0 || YYLAST < yyn || yycheck[yyn] != yytoken)
    goto yydefault;
  yyn = yytable[yyn];
  if (yyn <= 0)
    {
      if (yytable_value_is_error (yyn))
        goto yyerrlab;
      yyn = -yyn;
      goto yyreduce;
    }

  /* Count tokens shifted since error; after three, turn off error
     status.  */
  if (yyerrstatus)
    yyerrstatus--;

  /* Shift the lookahead token.  */
  YY_SYMBOL_PRINT ("Shifting", yytoken, &yylval, &yylloc);

  /* Discard the shifted token.  */
  yychar = YYEMPTY;

  yystate = yyn;
  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END
  *++yylsp = yylloc;
  goto yynewstate;


/*-----------------------------------------------------------.
| yydefault -- do the default action for the current state.  |
`-----------------------------------------------------------*/
yydefault:
  yyn = yydefact[yystate];
  if (yyn == 0)
    goto yyerrlab;
  goto yyreduce;


/*-----------------------------.
| yyreduce -- Do a reduction.  |
`-----------------------------*/
yyreduce:
  /* yyn is the number of a rule to reduce with.  */
  yylen = yyr2[yyn];

  /* If YYLEN is nonzero, implement the default value of the action:
     '$$ = $1'.

     Otherwise, the following line sets YYVAL to garbage.
     This behavior is undocumented and Bison
     users should not rely upon it.  Assigning to YYVAL
     unconditionally makes the parser a bit smaller, and it avoids a
     GCC warning that YYVAL may be used uninitialized.  */
  yyval = yyvsp[1-yylen];

  /* Default location. */
  YYLLOC_DEFAULT (yyloc, (yylsp - yylen), yylen);
  yyerror_range[1] = yyloc;
  YY_REDUCE_PRINT (yyn);
  switch (yyn)
    {
        case 2:
#line 289 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      _mesa_glsl_initialize_types(state);
   }
#line 2382 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 3:
#line 293 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      delete state->symbols;
      state->symbols = new(ralloc_parent(state)) glsl_symbol_table;
      if (state->es_shader) {
         if (state->stage == MESA_SHADER_FRAGMENT) {
            state->symbols->add_default_precision_qualifier("int", ast_precision_medium);
         } else {
            state->symbols->add_default_precision_qualifier("float", ast_precision_high);
            state->symbols->add_default_precision_qualifier("int", ast_precision_high);
         }
         state->symbols->add_default_precision_qualifier("sampler2D", ast_precision_low);
         state->symbols->add_default_precision_qualifier("samplerExternalOES", ast_precision_low);
         state->symbols->add_default_precision_qualifier("samplerCube", ast_precision_low);
         state->symbols->add_default_precision_qualifier("atomic_uint", ast_precision_high);
      }
      _mesa_glsl_initialize_types(state);
   }
#line 2404 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 5:
#line 315 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      state->process_version_directive(&(yylsp[-1]), (yyvsp[-1].n), NULL);
      if (state->error) {
         YYERROR;
      }
   }
#line 2415 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 6:
#line 322 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      state->process_version_directive(&(yylsp[-2]), (yyvsp[-2].n), (yyvsp[-1].identifier));
      if (state->error) {
         YYERROR;
      }
   }
#line 2426 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 11:
#line 336 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      /* Pragma invariant(all) cannot be used in a fragment shader.
       *
       * Page 27 of the GLSL 1.20 spec, Page 53 of the GLSL ES 3.00 spec:
       *
       *     "It is an error to use this pragma in a fragment shader."
       */
      if (state->is_version(120, 300) &&
          state->stage == MESA_SHADER_FRAGMENT) {
         _mesa_glsl_error(& (yylsp[-1]), state,
                          "pragma `invariant(all)' cannot be used "
                          "in a fragment shader.");
      } else if (!state->is_version(120, 100)) {
         _mesa_glsl_warning(& (yylsp[-1]), state,
                            "pragma `invariant(all)' not supported in %s "
                            "(GLSL ES 1.00 or GLSL 1.20 required)",
                            state->get_version_string());
      } else {
         state->all_invariant = true;
      }
   }
#line 2452 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 17:
#line 372 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      if (!_mesa_glsl_process_extension((yyvsp[-3].identifier), & (yylsp[-3]), (yyvsp[-1].identifier), & (yylsp[-1]), state)) {
         YYERROR;
      }
   }
#line 2462 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 18:
#line 381 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      /* FINISHME: The NULL test is required because pragmas are set to
       * FINISHME: NULL. (See production rule for external_declaration.)
       */
      if ((yyvsp[0].node) != NULL)
         state->translation_unit.push_tail(& (yyvsp[0].node)->link);
   }
#line 2474 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 19:
#line 389 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      /* FINISHME: The NULL test is required because pragmas are set to
       * FINISHME: NULL. (See production rule for external_declaration.)
       */
      if ((yyvsp[0].node) != NULL)
         state->translation_unit.push_tail(& (yyvsp[0].node)->link);
   }
#line 2486 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 20:
#line 396 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      if (!state->allow_extension_directive_midshader) {
         _mesa_glsl_error(& (yylsp[0]), state,
                          "#extension directive is not allowed "
                          "in the middle of a shader");
         YYERROR;
      }
   }
#line 2499 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 23:
#line 413 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      void *ctx = state->linalloc;
      (yyval.expression) = new(ctx) ast_expression(ast_identifier, NULL, NULL, NULL);
      (yyval.expression)->set_location((yylsp[0]));
      (yyval.expression)->primary_expression.identifier = (yyvsp[0].identifier);
   }
#line 2510 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 24:
#line 420 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      void *ctx = state->linalloc;
      (yyval.expression) = new(ctx) ast_expression(ast_int_constant, NULL, NULL, NULL);
      (yyval.expression)->set_location((yylsp[0]));
      (yyval.expression)->primary_expression.int_constant = (yyvsp[0].n);
   }
#line 2521 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 25:
#line 427 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      void *ctx = state->linalloc;
      (yyval.expression) = new(ctx) ast_expression(ast_uint_constant, NULL, NULL, NULL);
      (yyval.expression)->set_location((yylsp[0]));
      (yyval.expression)->primary_expression.uint_constant = (yyvsp[0].n);
   }
#line 2532 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 26:
#line 434 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      void *ctx = state->linalloc;
      (yyval.expression) = new(ctx) ast_expression(ast_int64_constant, NULL, NULL, NULL);
      (yyval.expression)->set_location((yylsp[0]));
      (yyval.expression)->primary_expression.int64_constant = (yyvsp[0].n64);
   }
#line 2543 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 27:
#line 441 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      void *ctx = state->linalloc;
      (yyval.expression) = new(ctx) ast_expression(ast_uint64_constant, NULL, NULL, NULL);
      (yyval.expression)->set_location((yylsp[0]));
      (yyval.expression)->primary_expression.uint64_constant = (yyvsp[0].n64);
   }
#line 2554 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 28:
#line 448 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      void *ctx = state->linalloc;
      (yyval.expression) = new(ctx) ast_expression(ast_float_constant, NULL, NULL, NULL);
      (yyval.expression)->set_location((yylsp[0]));
      (yyval.expression)->primary_expression.float_constant = (yyvsp[0].real);
   }
#line 2565 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 29:
#line 455 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      void *ctx = state->linalloc;
      (yyval.expression) = new(ctx) ast_expression(ast_double_constant, NULL, NULL, NULL);
      (yyval.expression)->set_location((yylsp[0]));
      (yyval.expression)->primary_expression.double_constant = (yyvsp[0].dreal);
   }
#line 2576 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 30:
#line 462 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      void *ctx = state->linalloc;
      (yyval.expression) = new(ctx) ast_expression(ast_bool_constant, NULL, NULL, NULL);
      (yyval.expression)->set_location((yylsp[0]));
      (yyval.expression)->primary_expression.bool_constant = (yyvsp[0].n);
   }
#line 2587 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 31:
#line 469 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      (yyval.expression) = (yyvsp[-1].expression);
   }
#line 2595 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 33:
#line 477 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      void *ctx = state->linalloc;
      (yyval.expression) = new(ctx) ast_expression(ast_array_index, (yyvsp[-3].expression), (yyvsp[-1].expression), NULL);
      (yyval.expression)->set_location_range((yylsp[-3]), (yylsp[0]));
   }
#line 2605 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 34:
#line 483 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      (yyval.expression) = (yyvsp[0].expression);
   }
#line 2613 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 35:
#line 487 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      void *ctx = state->linalloc;
      (yyval.expression) = new(ctx) ast_expression(ast_field_selection, (yyvsp[-2].expression), NULL, NULL);
      (yyval.expression)->set_location_range((yylsp[-2]), (yylsp[0]));
      (yyval.expression)->primary_expression.identifier = (yyvsp[0].identifier);
   }
#line 2624 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 36:
#line 494 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      void *ctx = state->linalloc;
      (yyval.expression) = new(ctx) ast_expression(ast_post_inc, (yyvsp[-1].expression), NULL, NULL);
      (yyval.expression)->set_location_range((yylsp[-1]), (yylsp[0]));
   }
#line 2634 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 37:
#line 500 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      void *ctx = state->linalloc;
      (yyval.expression) = new(ctx) ast_expression(ast_post_dec, (yyvsp[-1].expression), NULL, NULL);
      (yyval.expression)->set_location_range((yylsp[-1]), (yylsp[0]));
   }
#line 2644 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 45:
#line 531 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      (yyval.expression) = (yyvsp[-1].expression);
      (yyval.expression)->set_location((yylsp[-1]));
      (yyval.expression)->expressions.push_tail(& (yyvsp[0].expression)->link);
   }
#line 2654 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 46:
#line 537 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      (yyval.expression) = (yyvsp[-2].expression);
      (yyval.expression)->set_location((yylsp[-2]));
      (yyval.expression)->expressions.push_tail(& (yyvsp[0].expression)->link);
   }
#line 2664 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 48:
#line 553 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      void *ctx = state->linalloc;
      (yyval.expression) = new(ctx) ast_function_expression((yyvsp[0].type_specifier));
      (yyval.expression)->set_location((yylsp[0]));
      }
#line 2674 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 49:
#line 559 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      void *ctx = state->linalloc;
      (yyval.expression) = new(ctx) ast_function_expression((yyvsp[0].expression));
      (yyval.expression)->set_location((yylsp[0]));
      }
#line 2684 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 51:
#line 574 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      void *ctx = state->linalloc;
      (yyval.expression) = new(ctx) ast_expression(ast_pre_inc, (yyvsp[0].expression), NULL, NULL);
      (yyval.expression)->set_location((yylsp[-1]));
   }
#line 2694 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 52:
#line 580 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      void *ctx = state->linalloc;
      (yyval.expression) = new(ctx) ast_expression(ast_pre_dec, (yyvsp[0].expression), NULL, NULL);
      (yyval.expression)->set_location((yylsp[-1]));
   }
#line 2704 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 53:
#line 586 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      void *ctx = state->linalloc;
      (yyval.expression) = new(ctx) ast_expression((yyvsp[-1].n), (yyvsp[0].expression), NULL, NULL);
      (yyval.expression)->set_location_range((yylsp[-1]), (yylsp[0]));
   }
#line 2714 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 54:
#line 595 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    { (yyval.n) = ast_plus; }
#line 2720 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 55:
#line 596 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    { (yyval.n) = ast_neg; }
#line 2726 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 56:
#line 597 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    { (yyval.n) = ast_logic_not; }
#line 2732 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 57:
#line 598 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    { (yyval.n) = ast_bit_not; }
#line 2738 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 59:
#line 604 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      void *ctx = state->linalloc;
      (yyval.expression) = new(ctx) ast_expression_bin(ast_mul, (yyvsp[-2].expression), (yyvsp[0].expression));
      (yyval.expression)->set_location_range((yylsp[-2]), (yylsp[0]));
   }
#line 2748 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 60:
#line 610 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      void *ctx = state->linalloc;
      (yyval.expression) = new(ctx) ast_expression_bin(ast_div, (yyvsp[-2].expression), (yyvsp[0].expression));
      (yyval.expression)->set_location_range((yylsp[-2]), (yylsp[0]));
   }
#line 2758 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 61:
#line 616 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      void *ctx = state->linalloc;
      (yyval.expression) = new(ctx) ast_expression_bin(ast_mod, (yyvsp[-2].expression), (yyvsp[0].expression));
      (yyval.expression)->set_location_range((yylsp[-2]), (yylsp[0]));
   }
#line 2768 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 63:
#line 626 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      void *ctx = state->linalloc;
      (yyval.expression) = new(ctx) ast_expression_bin(ast_add, (yyvsp[-2].expression), (yyvsp[0].expression));
      (yyval.expression)->set_location_range((yylsp[-2]), (yylsp[0]));
   }
#line 2778 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 64:
#line 632 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      void *ctx = state->linalloc;
      (yyval.expression) = new(ctx) ast_expression_bin(ast_sub, (yyvsp[-2].expression), (yyvsp[0].expression));
      (yyval.expression)->set_location_range((yylsp[-2]), (yylsp[0]));
   }
#line 2788 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 66:
#line 642 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      void *ctx = state->linalloc;
      (yyval.expression) = new(ctx) ast_expression_bin(ast_lshift, (yyvsp[-2].expression), (yyvsp[0].expression));
      (yyval.expression)->set_location_range((yylsp[-2]), (yylsp[0]));
   }
#line 2798 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 67:
#line 648 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      void *ctx = state->linalloc;
      (yyval.expression) = new(ctx) ast_expression_bin(ast_rshift, (yyvsp[-2].expression), (yyvsp[0].expression));
      (yyval.expression)->set_location_range((yylsp[-2]), (yylsp[0]));
   }
#line 2808 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 69:
#line 658 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      void *ctx = state->linalloc;
      (yyval.expression) = new(ctx) ast_expression_bin(ast_less, (yyvsp[-2].expression), (yyvsp[0].expression));
      (yyval.expression)->set_location_range((yylsp[-2]), (yylsp[0]));
   }
#line 2818 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 70:
#line 664 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      void *ctx = state->linalloc;
      (yyval.expression) = new(ctx) ast_expression_bin(ast_greater, (yyvsp[-2].expression), (yyvsp[0].expression));
      (yyval.expression)->set_location_range((yylsp[-2]), (yylsp[0]));
   }
#line 2828 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 71:
#line 670 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      void *ctx = state->linalloc;
      (yyval.expression) = new(ctx) ast_expression_bin(ast_lequal, (yyvsp[-2].expression), (yyvsp[0].expression));
      (yyval.expression)->set_location_range((yylsp[-2]), (yylsp[0]));
   }
#line 2838 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 72:
#line 676 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      void *ctx = state->linalloc;
      (yyval.expression) = new(ctx) ast_expression_bin(ast_gequal, (yyvsp[-2].expression), (yyvsp[0].expression));
      (yyval.expression)->set_location_range((yylsp[-2]), (yylsp[0]));
   }
#line 2848 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 74:
#line 686 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      void *ctx = state->linalloc;
      (yyval.expression) = new(ctx) ast_expression_bin(ast_equal, (yyvsp[-2].expression), (yyvsp[0].expression));
      (yyval.expression)->set_location_range((yylsp[-2]), (yylsp[0]));
   }
#line 2858 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 75:
#line 692 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      void *ctx = state->linalloc;
      (yyval.expression) = new(ctx) ast_expression_bin(ast_nequal, (yyvsp[-2].expression), (yyvsp[0].expression));
      (yyval.expression)->set_location_range((yylsp[-2]), (yylsp[0]));
   }
#line 2868 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 77:
#line 702 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      void *ctx = state->linalloc;
      (yyval.expression) = new(ctx) ast_expression_bin(ast_bit_and, (yyvsp[-2].expression), (yyvsp[0].expression));
      (yyval.expression)->set_location_range((yylsp[-2]), (yylsp[0]));
   }
#line 2878 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 79:
#line 712 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      void *ctx = state->linalloc;
      (yyval.expression) = new(ctx) ast_expression_bin(ast_bit_xor, (yyvsp[-2].expression), (yyvsp[0].expression));
      (yyval.expression)->set_location_range((yylsp[-2]), (yylsp[0]));
   }
#line 2888 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 81:
#line 722 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      void *ctx = state->linalloc;
      (yyval.expression) = new(ctx) ast_expression_bin(ast_bit_or, (yyvsp[-2].expression), (yyvsp[0].expression));
      (yyval.expression)->set_location_range((yylsp[-2]), (yylsp[0]));
   }
#line 2898 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 83:
#line 732 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      void *ctx = state->linalloc;
      (yyval.expression) = new(ctx) ast_expression_bin(ast_logic_and, (yyvsp[-2].expression), (yyvsp[0].expression));
      (yyval.expression)->set_location_range((yylsp[-2]), (yylsp[0]));
   }
#line 2908 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 85:
#line 742 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      void *ctx = state->linalloc;
      (yyval.expression) = new(ctx) ast_expression_bin(ast_logic_xor, (yyvsp[-2].expression), (yyvsp[0].expression));
      (yyval.expression)->set_location_range((yylsp[-2]), (yylsp[0]));
   }
#line 2918 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 87:
#line 752 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      void *ctx = state->linalloc;
      (yyval.expression) = new(ctx) ast_expression_bin(ast_logic_or, (yyvsp[-2].expression), (yyvsp[0].expression));
      (yyval.expression)->set_location_range((yylsp[-2]), (yylsp[0]));
   }
#line 2928 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 89:
#line 762 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      void *ctx = state->linalloc;
      (yyval.expression) = new(ctx) ast_expression(ast_conditional, (yyvsp[-4].expression), (yyvsp[-2].expression), (yyvsp[0].expression));
      (yyval.expression)->set_location_range((yylsp[-4]), (yylsp[0]));
   }
#line 2938 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 91:
#line 772 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      void *ctx = state->linalloc;
      (yyval.expression) = new(ctx) ast_expression((yyvsp[-1].n), (yyvsp[-2].expression), (yyvsp[0].expression), NULL);
      (yyval.expression)->set_location_range((yylsp[-2]), (yylsp[0]));
   }
#line 2948 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 92:
#line 780 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    { (yyval.n) = ast_assign; }
#line 2954 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 93:
#line 781 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    { (yyval.n) = ast_mul_assign; }
#line 2960 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 94:
#line 782 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    { (yyval.n) = ast_div_assign; }
#line 2966 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 95:
#line 783 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    { (yyval.n) = ast_mod_assign; }
#line 2972 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 96:
#line 784 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    { (yyval.n) = ast_add_assign; }
#line 2978 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 97:
#line 785 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    { (yyval.n) = ast_sub_assign; }
#line 2984 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 98:
#line 786 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    { (yyval.n) = ast_ls_assign; }
#line 2990 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 99:
#line 787 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    { (yyval.n) = ast_rs_assign; }
#line 2996 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 100:
#line 788 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    { (yyval.n) = ast_and_assign; }
#line 3002 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 101:
#line 789 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    { (yyval.n) = ast_xor_assign; }
#line 3008 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 102:
#line 790 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    { (yyval.n) = ast_or_assign; }
#line 3014 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 103:
#line 795 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      (yyval.expression) = (yyvsp[0].expression);
   }
#line 3022 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 104:
#line 799 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      void *ctx = state->linalloc;
      if ((yyvsp[-2].expression)->oper != ast_sequence) {
         (yyval.expression) = new(ctx) ast_expression(ast_sequence, NULL, NULL, NULL);
         (yyval.expression)->set_location_range((yylsp[-2]), (yylsp[0]));
         (yyval.expression)->expressions.push_tail(& (yyvsp[-2].expression)->link);
      } else {
         (yyval.expression) = (yyvsp[-2].expression);
      }

      (yyval.expression)->expressions.push_tail(& (yyvsp[0].expression)->link);
   }
#line 3039 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 106:
#line 819 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      state->symbols->pop_scope();
      (yyval.node) = (yyvsp[-1].function);
   }
#line 3048 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 107:
#line 824 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      (yyval.node) = (yyvsp[-1].declarator_list);
   }
#line 3056 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 108:
#line 828 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      (yyvsp[-1].type_specifier)->default_precision = (yyvsp[-2].n);
      (yyval.node) = (yyvsp[-1].type_specifier);
   }
#line 3065 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 109:
#line 833 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      ast_interface_block *block = (ast_interface_block *) (yyvsp[0].node);
      if (block->layout.has_layout() || block->layout.has_memory()) {
         if (!block->default_layout.merge_qualifier(& (yylsp[0]), state, block->layout, false)) {
            YYERROR;
         }
      }
      block->layout = block->default_layout;
      if (!block->layout.push_to_global(& (yylsp[0]), state)) {
         YYERROR;
      }
      (yyval.node) = (yyvsp[0].node);
   }
#line 3083 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 113:
#line 859 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      (yyval.function) = (yyvsp[-1].function);
      (yyval.function)->parameters.push_tail(& (yyvsp[0].parameter_declarator)->link);
   }
#line 3092 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 114:
#line 864 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      (yyval.function) = (yyvsp[-2].function);
      (yyval.function)->parameters.push_tail(& (yyvsp[0].parameter_declarator)->link);
   }
#line 3101 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 115:
#line 872 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      void *ctx = state->linalloc;
      (yyval.function) = new(ctx) ast_function();
      (yyval.function)->set_location((yylsp[-1]));
      (yyval.function)->return_type = (yyvsp[-2].fully_specified_type);
      (yyval.function)->identifier = (yyvsp[-1].identifier);

      if ((yyvsp[-2].fully_specified_type)->qualifier.is_subroutine_decl()) {
         /* add type for IDENTIFIER search */
         state->symbols->add_type((yyvsp[-1].identifier), glsl_type::get_subroutine_instance((yyvsp[-1].identifier)));
      } else
         state->symbols->add_function(new(state) ir_function((yyvsp[-1].identifier)));
      state->symbols->push_scope();
   }
#line 3120 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 116:
#line 890 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      void *ctx = state->linalloc;
      (yyval.parameter_declarator) = new(ctx) ast_parameter_declarator();
      (yyval.parameter_declarator)->set_location_range((yylsp[-1]), (yylsp[0]));
      (yyval.parameter_declarator)->type = new(ctx) ast_fully_specified_type();
      (yyval.parameter_declarator)->type->set_location((yylsp[-1]));
      (yyval.parameter_declarator)->type->specifier = (yyvsp[-1].type_specifier);
      (yyval.parameter_declarator)->identifier = (yyvsp[0].identifier);
      state->symbols->add_variable(new(state) ir_variable(NULL, (yyvsp[0].identifier), ir_var_auto));
   }
#line 3135 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 117:
#line 901 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      if (state->allow_layout_qualifier_on_function_parameter) {
         void *ctx = state->linalloc;
         (yyval.parameter_declarator) = new(ctx) ast_parameter_declarator();
         (yyval.parameter_declarator)->set_location_range((yylsp[-1]), (yylsp[0]));
         (yyval.parameter_declarator)->type = new(ctx) ast_fully_specified_type();
         (yyval.parameter_declarator)->type->set_location((yylsp[-1]));
         (yyval.parameter_declarator)->type->specifier = (yyvsp[-1].type_specifier);
         (yyval.parameter_declarator)->identifier = (yyvsp[0].identifier);
         state->symbols->add_variable(new(state) ir_variable(NULL, (yyvsp[0].identifier), ir_var_auto));
      } else {
         _mesa_glsl_error(&(yylsp[-2]), state,
                          "is is not allowed on function parameter");
         YYERROR;
      }
   }
#line 3156 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 118:
#line 918 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      void *ctx = state->linalloc;
      (yyval.parameter_declarator) = new(ctx) ast_parameter_declarator();
      (yyval.parameter_declarator)->set_location_range((yylsp[-2]), (yylsp[0]));
      (yyval.parameter_declarator)->type = new(ctx) ast_fully_specified_type();
      (yyval.parameter_declarator)->type->set_location((yylsp[-2]));
      (yyval.parameter_declarator)->type->specifier = (yyvsp[-2].type_specifier);
      (yyval.parameter_declarator)->identifier = (yyvsp[-1].identifier);
      (yyval.parameter_declarator)->array_specifier = (yyvsp[0].array_specifier);
      state->symbols->add_variable(new(state) ir_variable(NULL, (yyvsp[-1].identifier), ir_var_auto));
   }
#line 3172 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 119:
#line 933 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      (yyval.parameter_declarator) = (yyvsp[0].parameter_declarator);
      (yyval.parameter_declarator)->type->qualifier = (yyvsp[-1].type_qualifier);
      if (!(yyval.parameter_declarator)->type->qualifier.push_to_global(& (yylsp[-1]), state)) {
         YYERROR;
      }
   }
#line 3184 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 120:
#line 941 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      void *ctx = state->linalloc;
      (yyval.parameter_declarator) = new(ctx) ast_parameter_declarator();
      (yyval.parameter_declarator)->set_location((yylsp[0]));
      (yyval.parameter_declarator)->type = new(ctx) ast_fully_specified_type();
      (yyval.parameter_declarator)->type->set_location_range((yylsp[-1]), (yylsp[0]));
      (yyval.parameter_declarator)->type->qualifier = (yyvsp[-1].type_qualifier);
      if (!(yyval.parameter_declarator)->type->qualifier.push_to_global(& (yylsp[-1]), state)) {
         YYERROR;
      }
      (yyval.parameter_declarator)->type->specifier = (yyvsp[0].type_specifier);
   }
#line 3201 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 121:
#line 957 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      memset(& (yyval.type_qualifier), 0, sizeof((yyval.type_qualifier)));
   }
#line 3209 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 122:
#line 961 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      if ((yyvsp[0].type_qualifier).flags.q.constant)
         _mesa_glsl_error(&(yylsp[-1]), state, "duplicate const qualifier");

      (yyval.type_qualifier) = (yyvsp[0].type_qualifier);
      (yyval.type_qualifier).flags.q.constant = 1;
   }
#line 3221 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 123:
#line 969 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      if ((yyvsp[0].type_qualifier).flags.q.precise)
         _mesa_glsl_error(&(yylsp[-1]), state, "duplicate precise qualifier");

      (yyval.type_qualifier) = (yyvsp[0].type_qualifier);
      (yyval.type_qualifier).flags.q.precise = 1;
   }
#line 3233 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 124:
#line 977 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      if (((yyvsp[-1].type_qualifier).flags.q.in || (yyvsp[-1].type_qualifier).flags.q.out) && ((yyvsp[0].type_qualifier).flags.q.in || (yyvsp[0].type_qualifier).flags.q.out))
         _mesa_glsl_error(&(yylsp[-1]), state, "duplicate in/out/inout qualifier");

      if (!state->has_420pack_or_es31() && (yyvsp[0].type_qualifier).flags.q.constant)
         _mesa_glsl_error(&(yylsp[-1]), state, "in/out/inout must come after const "
                                      "or precise");

      (yyval.type_qualifier) = (yyvsp[-1].type_qualifier);
      (yyval.type_qualifier).merge_qualifier(&(yylsp[-1]), state, (yyvsp[0].type_qualifier), false);
   }
#line 3249 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 125:
#line 989 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      if ((yyvsp[0].type_qualifier).precision != ast_precision_none)
         _mesa_glsl_error(&(yylsp[-1]), state, "duplicate precision qualifier");

      if (!state->has_420pack_or_es31() &&
          (yyvsp[0].type_qualifier).flags.i != 0)
         _mesa_glsl_error(&(yylsp[-1]), state, "precision qualifiers must come last");

      (yyval.type_qualifier) = (yyvsp[0].type_qualifier);
      (yyval.type_qualifier).precision = (yyvsp[-1].n);
   }
#line 3265 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 126:
#line 1001 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      (yyval.type_qualifier) = (yyvsp[-1].type_qualifier);
      (yyval.type_qualifier).merge_qualifier(&(yylsp[-1]), state, (yyvsp[0].type_qualifier), false);
   }
#line 3274 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 127:
#line 1008 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      memset(& (yyval.type_qualifier), 0, sizeof((yyval.type_qualifier)));
      (yyval.type_qualifier).flags.q.in = 1;
   }
#line 3283 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 128:
#line 1013 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      memset(& (yyval.type_qualifier), 0, sizeof((yyval.type_qualifier)));
      (yyval.type_qualifier).flags.q.out = 1;
   }
#line 3292 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 129:
#line 1018 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      memset(& (yyval.type_qualifier), 0, sizeof((yyval.type_qualifier)));
      (yyval.type_qualifier).flags.q.in = 1;
      (yyval.type_qualifier).flags.q.out = 1;
   }
#line 3302 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 132:
#line 1032 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      void *ctx = state->linalloc;
      ast_declaration *decl = new(ctx) ast_declaration((yyvsp[0].identifier), NULL, NULL);
      decl->set_location((yylsp[0]));

      (yyval.declarator_list) = (yyvsp[-2].declarator_list);
      (yyval.declarator_list)->declarations.push_tail(&decl->link);
      state->symbols->add_variable(new(state) ir_variable(NULL, (yyvsp[0].identifier), ir_var_auto));
   }
#line 3316 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 133:
#line 1042 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      void *ctx = state->linalloc;
      ast_declaration *decl = new(ctx) ast_declaration((yyvsp[-1].identifier), (yyvsp[0].array_specifier), NULL);
      decl->set_location_range((yylsp[-1]), (yylsp[0]));

      (yyval.declarator_list) = (yyvsp[-3].declarator_list);
      (yyval.declarator_list)->declarations.push_tail(&decl->link);
      state->symbols->add_variable(new(state) ir_variable(NULL, (yyvsp[-1].identifier), ir_var_auto));
   }
#line 3330 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 134:
#line 1052 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      void *ctx = state->linalloc;
      ast_declaration *decl = new(ctx) ast_declaration((yyvsp[-3].identifier), (yyvsp[-2].array_specifier), (yyvsp[0].expression));
      decl->set_location_range((yylsp[-3]), (yylsp[-2]));

      (yyval.declarator_list) = (yyvsp[-5].declarator_list);
      (yyval.declarator_list)->declarations.push_tail(&decl->link);
      state->symbols->add_variable(new(state) ir_variable(NULL, (yyvsp[-3].identifier), ir_var_auto));
   }
#line 3344 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 135:
#line 1062 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      void *ctx = state->linalloc;
      ast_declaration *decl = new(ctx) ast_declaration((yyvsp[-2].identifier), NULL, (yyvsp[0].expression));
      decl->set_location((yylsp[-2]));

      (yyval.declarator_list) = (yyvsp[-4].declarator_list);
      (yyval.declarator_list)->declarations.push_tail(&decl->link);
      state->symbols->add_variable(new(state) ir_variable(NULL, (yyvsp[-2].identifier), ir_var_auto));
   }
#line 3358 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 136:
#line 1076 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      void *ctx = state->linalloc;
      /* Empty declaration list is valid. */
      (yyval.declarator_list) = new(ctx) ast_declarator_list((yyvsp[0].fully_specified_type));
      (yyval.declarator_list)->set_location((yylsp[0]));
   }
#line 3369 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 137:
#line 1083 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      void *ctx = state->linalloc;
      ast_declaration *decl = new(ctx) ast_declaration((yyvsp[0].identifier), NULL, NULL);
      decl->set_location((yylsp[0]));

      (yyval.declarator_list) = new(ctx) ast_declarator_list((yyvsp[-1].fully_specified_type));
      (yyval.declarator_list)->set_location_range((yylsp[-1]), (yylsp[0]));
      (yyval.declarator_list)->declarations.push_tail(&decl->link);
      state->symbols->add_variable(new(state) ir_variable(NULL, (yyvsp[0].identifier), ir_var_auto));
   }
#line 3384 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 138:
#line 1094 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      void *ctx = state->linalloc;
      ast_declaration *decl = new(ctx) ast_declaration((yyvsp[-1].identifier), (yyvsp[0].array_specifier), NULL);
      decl->set_location_range((yylsp[-1]), (yylsp[0]));

      (yyval.declarator_list) = new(ctx) ast_declarator_list((yyvsp[-2].fully_specified_type));
      (yyval.declarator_list)->set_location_range((yylsp[-2]), (yylsp[0]));
      (yyval.declarator_list)->declarations.push_tail(&decl->link);
      state->symbols->add_variable(new(state) ir_variable(NULL, (yyvsp[-1].identifier), ir_var_auto));
   }
#line 3399 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 139:
#line 1105 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      void *ctx = state->linalloc;
      ast_declaration *decl = new(ctx) ast_declaration((yyvsp[-3].identifier), (yyvsp[-2].array_specifier), (yyvsp[0].expression));
      decl->set_location_range((yylsp[-3]), (yylsp[-2]));

      (yyval.declarator_list) = new(ctx) ast_declarator_list((yyvsp[-4].fully_specified_type));
      (yyval.declarator_list)->set_location_range((yylsp[-4]), (yylsp[-2]));
      (yyval.declarator_list)->declarations.push_tail(&decl->link);
      state->symbols->add_variable(new(state) ir_variable(NULL, (yyvsp[-3].identifier), ir_var_auto));
   }
#line 3414 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 140:
#line 1116 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      void *ctx = state->linalloc;
      ast_declaration *decl = new(ctx) ast_declaration((yyvsp[-2].identifier), NULL, (yyvsp[0].expression));
      decl->set_location((yylsp[-2]));

      (yyval.declarator_list) = new(ctx) ast_declarator_list((yyvsp[-3].fully_specified_type));
      (yyval.declarator_list)->set_location_range((yylsp[-3]), (yylsp[-2]));
      (yyval.declarator_list)->declarations.push_tail(&decl->link);
      state->symbols->add_variable(new(state) ir_variable(NULL, (yyvsp[-2].identifier), ir_var_auto));
   }
#line 3429 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 141:
#line 1127 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      void *ctx = state->linalloc;
      ast_declaration *decl = new(ctx) ast_declaration((yyvsp[0].identifier), NULL, NULL);
      decl->set_location((yylsp[0]));

      (yyval.declarator_list) = new(ctx) ast_declarator_list(NULL);
      (yyval.declarator_list)->set_location_range((yylsp[-1]), (yylsp[0]));
      (yyval.declarator_list)->invariant = true;

      (yyval.declarator_list)->declarations.push_tail(&decl->link);
   }
#line 3445 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 142:
#line 1139 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      void *ctx = state->linalloc;
      ast_declaration *decl = new(ctx) ast_declaration((yyvsp[0].identifier), NULL, NULL);
      decl->set_location((yylsp[0]));

      (yyval.declarator_list) = new(ctx) ast_declarator_list(NULL);
      (yyval.declarator_list)->set_location_range((yylsp[-1]), (yylsp[0]));
      (yyval.declarator_list)->precise = true;

      (yyval.declarator_list)->declarations.push_tail(&decl->link);
   }
#line 3461 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 143:
#line 1154 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      void *ctx = state->linalloc;
      (yyval.fully_specified_type) = new(ctx) ast_fully_specified_type();
      (yyval.fully_specified_type)->set_location((yylsp[0]));
      (yyval.fully_specified_type)->specifier = (yyvsp[0].type_specifier);
   }
#line 3472 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 144:
#line 1161 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      void *ctx = state->linalloc;
      (yyval.fully_specified_type) = new(ctx) ast_fully_specified_type();
      (yyval.fully_specified_type)->set_location_range((yylsp[-1]), (yylsp[0]));
      (yyval.fully_specified_type)->qualifier = (yyvsp[-1].type_qualifier);
      if (!(yyval.fully_specified_type)->qualifier.push_to_global(& (yylsp[-1]), state)) {
         YYERROR;
      }
      (yyval.fully_specified_type)->specifier = (yyvsp[0].type_specifier);
      if ((yyval.fully_specified_type)->specifier->structure != NULL &&
          (yyval.fully_specified_type)->specifier->structure->is_declaration) {
            (yyval.fully_specified_type)->specifier->structure->layout = &(yyval.fully_specified_type)->qualifier;
      }
   }
#line 3491 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 145:
#line 1179 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      (yyval.type_qualifier) = (yyvsp[-1].type_qualifier);
   }
#line 3499 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 147:
#line 1187 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      (yyval.type_qualifier) = (yyvsp[-2].type_qualifier);
      if (!(yyval.type_qualifier).merge_qualifier(& (yylsp[0]), state, (yyvsp[0].type_qualifier), true)) {
         YYERROR;
      }
   }
#line 3510 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 148:
#line 1197 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      memset(& (yyval.type_qualifier), 0, sizeof((yyval.type_qualifier)));

      /* Layout qualifiers for ARB_fragment_coord_conventions. */
      if (!(yyval.type_qualifier).flags.i && (state->ARB_fragment_coord_conventions_enable ||
                          state->is_version(150, 0))) {
         if (match_layout_qualifier((yyvsp[0].identifier), "origin_upper_left", state) == 0) {
            (yyval.type_qualifier).flags.q.origin_upper_left = 1;
         } else if (match_layout_qualifier((yyvsp[0].identifier), "pixel_center_integer",
                                           state) == 0) {
            (yyval.type_qualifier).flags.q.pixel_center_integer = 1;
         }

         if ((yyval.type_qualifier).flags.i && state->ARB_fragment_coord_conventions_warn) {
            _mesa_glsl_warning(& (yylsp[0]), state,
                               "GL_ARB_fragment_coord_conventions layout "
                               "identifier `%s' used", (yyvsp[0].identifier));
         }
      }

      /* Layout qualifiers for AMD/ARB_conservative_depth. */
      if (!(yyval.type_qualifier).flags.i &&
          (state->AMD_conservative_depth_enable ||
           state->ARB_conservative_depth_enable ||
           state->is_version(420, 0))) {
         if (match_layout_qualifier((yyvsp[0].identifier), "depth_any", state) == 0) {
            (yyval.type_qualifier).flags.q.depth_type = 1;
            (yyval.type_qualifier).depth_type = ast_depth_any;
         } else if (match_layout_qualifier((yyvsp[0].identifier), "depth_greater", state) == 0) {
            (yyval.type_qualifier).flags.q.depth_type = 1;
            (yyval.type_qualifier).depth_type = ast_depth_greater;
         } else if (match_layout_qualifier((yyvsp[0].identifier), "depth_less", state) == 0) {
            (yyval.type_qualifier).flags.q.depth_type = 1;
            (yyval.type_qualifier).depth_type = ast_depth_less;
         } else if (match_layout_qualifier((yyvsp[0].identifier), "depth_unchanged",
                                           state) == 0) {
            (yyval.type_qualifier).flags.q.depth_type = 1;
            (yyval.type_qualifier).depth_type = ast_depth_unchanged;
         }

         if ((yyval.type_qualifier).flags.i && state->AMD_conservative_depth_warn) {
            _mesa_glsl_warning(& (yylsp[0]), state,
                               "GL_AMD_conservative_depth "
                               "layout qualifier `%s' is used", (yyvsp[0].identifier));
         }
         if ((yyval.type_qualifier).flags.i && state->ARB_conservative_depth_warn) {
            _mesa_glsl_warning(& (yylsp[0]), state,
                               "GL_ARB_conservative_depth "
                               "layout qualifier `%s' is used", (yyvsp[0].identifier));
         }
      }

      /* See also interface_block_layout_qualifier. */
      if (!(yyval.type_qualifier).flags.i && state->has_uniform_buffer_objects()) {
         if (match_layout_qualifier((yyvsp[0].identifier), "std140", state) == 0) {
            (yyval.type_qualifier).flags.q.std140 = 1;
         } else if (match_layout_qualifier((yyvsp[0].identifier), "shared", state) == 0) {
            (yyval.type_qualifier).flags.q.shared = 1;
         } else if (match_layout_qualifier((yyvsp[0].identifier), "std430", state) == 0) {
            (yyval.type_qualifier).flags.q.std430 = 1;
         } else if (match_layout_qualifier((yyvsp[0].identifier), "column_major", state) == 0) {
            (yyval.type_qualifier).flags.q.column_major = 1;
         /* "row_major" is a reserved word in GLSL 1.30+. Its token is parsed
          * below in the interface_block_layout_qualifier rule.
          *
          * It is not a reserved word in GLSL ES 3.00, so it's handled here as
          * an identifier.
          *
          * Also, this takes care of alternate capitalizations of
          * "row_major" (which is necessary because layout qualifiers
          * are case-insensitive in desktop GLSL).
          */
         } else if (match_layout_qualifier((yyvsp[0].identifier), "row_major", state) == 0) {
            (yyval.type_qualifier).flags.q.row_major = 1;
         /* "packed" is a reserved word in GLSL, and its token is
          * parsed below in the interface_block_layout_qualifier rule.
          * However, we must take care of alternate capitalizations of
          * "packed", because layout qualifiers are case-insensitive
          * in desktop GLSL.
          */
         } else if (match_layout_qualifier((yyvsp[0].identifier), "packed", state) == 0) {
           (yyval.type_qualifier).flags.q.packed = 1;
         }

         if ((yyval.type_qualifier).flags.i && state->ARB_uniform_buffer_object_warn) {
            _mesa_glsl_warning(& (yylsp[0]), state,
                               "#version 140 / GL_ARB_uniform_buffer_object "
                               "layout qualifier `%s' is used", (yyvsp[0].identifier));
         }
      }

      /* Layout qualifiers for GLSL 1.50 geometry shaders. */
      if (!(yyval.type_qualifier).flags.i) {
         static const struct {
            const char *s;
            GLenum e;
         } map[] = {
                 { "points", GL_POINTS },
                 { "lines", GL_LINES },
                 { "lines_adjacency", GL_LINES_ADJACENCY },
                 { "line_strip", GL_LINE_STRIP },
                 { "triangles", GL_TRIANGLES },
                 { "triangles_adjacency", GL_TRIANGLES_ADJACENCY },
                 { "triangle_strip", GL_TRIANGLE_STRIP },
         };
         for (unsigned i = 0; i < ARRAY_SIZE(map); i++) {
            if (match_layout_qualifier((yyvsp[0].identifier), map[i].s, state) == 0) {
               (yyval.type_qualifier).flags.q.prim_type = 1;
               (yyval.type_qualifier).prim_type = map[i].e;
               break;
            }
         }

         if ((yyval.type_qualifier).flags.i && !state->has_geometry_shader() &&
             !state->has_tessellation_shader()) {
            _mesa_glsl_error(& (yylsp[0]), state, "#version 150 layout "
                             "qualifier `%s' used", (yyvsp[0].identifier));
         }
      }

      /* Layout qualifiers for ARB_shader_image_load_store. */
      if (state->has_shader_image_load_store()) {
         if (!(yyval.type_qualifier).flags.i) {
            static const struct {
               const char *name;
               GLenum format;
               glsl_base_type base_type;
               /** Minimum desktop GLSL version required for the image
                * format.  Use 130 if already present in the original
                * ARB extension.
                */
               unsigned required_glsl;
               /** Minimum GLSL ES version required for the image format. */
               unsigned required_essl;
               /* NV_image_formats */
               bool nv_image_formats;
            } map[] = {
               { "rgba32f", GL_RGBA32F, GLSL_TYPE_FLOAT, 130, 310, false },
               { "rgba16f", GL_RGBA16F, GLSL_TYPE_FLOAT, 130, 310, false },
               { "rg32f", GL_RG32F, GLSL_TYPE_FLOAT, 130, 0, true },
               { "rg16f", GL_RG16F, GLSL_TYPE_FLOAT, 130, 0, true },
               { "r11f_g11f_b10f", GL_R11F_G11F_B10F, GLSL_TYPE_FLOAT, 130, 0, true },
               { "r32f", GL_R32F, GLSL_TYPE_FLOAT, 130, 310, false },
               { "r16f", GL_R16F, GLSL_TYPE_FLOAT, 130, 0, true },
               { "rgba32ui", GL_RGBA32UI, GLSL_TYPE_UINT, 130, 310, false },
               { "rgba16ui", GL_RGBA16UI, GLSL_TYPE_UINT, 130, 310, false },
               { "rgb10_a2ui", GL_RGB10_A2UI, GLSL_TYPE_UINT, 130, 0, true },
               { "rgba8ui", GL_RGBA8UI, GLSL_TYPE_UINT, 130, 310, false },
               { "rg32ui", GL_RG32UI, GLSL_TYPE_UINT, 130, 0, true },
               { "rg16ui", GL_RG16UI, GLSL_TYPE_UINT, 130, 0, true },
               { "rg8ui", GL_RG8UI, GLSL_TYPE_UINT, 130, 0, true },
               { "r32ui", GL_R32UI, GLSL_TYPE_UINT, 130, 310, false },
               { "r16ui", GL_R16UI, GLSL_TYPE_UINT, 130, 0, true },
               { "r8ui", GL_R8UI, GLSL_TYPE_UINT, 130, 0, true },
               { "rgba32i", GL_RGBA32I, GLSL_TYPE_INT, 130, 310, false },
               { "rgba16i", GL_RGBA16I, GLSL_TYPE_INT, 130, 310, false },
               { "rgba8i", GL_RGBA8I, GLSL_TYPE_INT, 130, 310, false },
               { "rg32i", GL_RG32I, GLSL_TYPE_INT, 130, 0, true },
               { "rg16i", GL_RG16I, GLSL_TYPE_INT, 130, 0, true },
               { "rg8i", GL_RG8I, GLSL_TYPE_INT, 130, 0, true },
               { "r32i", GL_R32I, GLSL_TYPE_INT, 130, 310, false },
               { "r16i", GL_R16I, GLSL_TYPE_INT, 130, 0, true },
               { "r8i", GL_R8I, GLSL_TYPE_INT, 130, 0, true },
               { "rgba16", GL_RGBA16, GLSL_TYPE_FLOAT, 130, 0, true },
               { "rgb10_a2", GL_RGB10_A2, GLSL_TYPE_FLOAT, 130, 0, true },
               { "rgba8", GL_RGBA8, GLSL_TYPE_FLOAT, 130, 310, false },
               { "rg16", GL_RG16, GLSL_TYPE_FLOAT, 130, 0, true },
               { "rg8", GL_RG8, GLSL_TYPE_FLOAT, 130, 0, true },
               { "r16", GL_R16, GLSL_TYPE_FLOAT, 130, 0, true },
               { "r8", GL_R8, GLSL_TYPE_FLOAT, 130, 0, true },
               { "rgba16_snorm", GL_RGBA16_SNORM, GLSL_TYPE_FLOAT, 130, 0, true },
               { "rgba8_snorm", GL_RGBA8_SNORM, GLSL_TYPE_FLOAT, 130, 310, false },
               { "rg16_snorm", GL_RG16_SNORM, GLSL_TYPE_FLOAT, 130, 0, true },
               { "rg8_snorm", GL_RG8_SNORM, GLSL_TYPE_FLOAT, 130, 0, true },
               { "r16_snorm", GL_R16_SNORM, GLSL_TYPE_FLOAT, 130, 0, true },
               { "r8_snorm", GL_R8_SNORM, GLSL_TYPE_FLOAT, 130, 0, true }
            };

            for (unsigned i = 0; i < ARRAY_SIZE(map); i++) {
               if ((state->is_version(map[i].required_glsl,
                                      map[i].required_essl) ||
                    (state->NV_image_formats_enable &&
                     map[i].nv_image_formats)) &&
                   match_layout_qualifier((yyvsp[0].identifier), map[i].name, state) == 0) {
                  (yyval.type_qualifier).flags.q.explicit_image_format = 1;
                  (yyval.type_qualifier).image_format = map[i].format;
                  (yyval.type_qualifier).image_base_type = map[i].base_type;
                  break;
               }
            }
         }
      }

      if (!(yyval.type_qualifier).flags.i) {
         if (match_layout_qualifier((yyvsp[0].identifier), "early_fragment_tests", state) == 0) {
            /* From section 4.4.1.3 of the GLSL 4.50 specification
             * (Fragment Shader Inputs):
             *
             *  "Fragment shaders also allow the following layout
             *   qualifier on in only (not with variable declarations)
             *     layout-qualifier-id
             *        early_fragment_tests
             *   [...]"
             */
            if (state->stage != MESA_SHADER_FRAGMENT) {
               _mesa_glsl_error(& (yylsp[0]), state,
                                "early_fragment_tests layout qualifier only "
                                "valid in fragment shaders");
            }

            (yyval.type_qualifier).flags.q.early_fragment_tests = 1;
         }

         if (match_layout_qualifier((yyvsp[0].identifier), "inner_coverage", state) == 0) {
            if (state->stage != MESA_SHADER_FRAGMENT) {
               _mesa_glsl_error(& (yylsp[0]), state,
                                "inner_coverage layout qualifier only "
                                "valid in fragment shaders");
            }

	    if (state->INTEL_conservative_rasterization_enable) {
	       (yyval.type_qualifier).flags.q.inner_coverage = 1;
	    } else {
	       _mesa_glsl_error(& (yylsp[0]), state,
                                "inner_coverage layout qualifier present, "
                                "but the INTEL_conservative_rasterization extension "
                                "is not enabled.");
            }
         }

         if (match_layout_qualifier((yyvsp[0].identifier), "post_depth_coverage", state) == 0) {
            if (state->stage != MESA_SHADER_FRAGMENT) {
               _mesa_glsl_error(& (yylsp[0]), state,
                                "post_depth_coverage layout qualifier only "
                                "valid in fragment shaders");
            }

            if (state->ARB_post_depth_coverage_enable ||
		state->INTEL_conservative_rasterization_enable) {
               (yyval.type_qualifier).flags.q.post_depth_coverage = 1;
            } else {
               _mesa_glsl_error(& (yylsp[0]), state,
                                "post_depth_coverage layout qualifier present, "
                                "but the GL_ARB_post_depth_coverage extension "
                                "is not enabled.");
            }
         }

         if ((yyval.type_qualifier).flags.q.post_depth_coverage && (yyval.type_qualifier).flags.q.inner_coverage) {
            _mesa_glsl_error(& (yylsp[0]), state,
                             "post_depth_coverage & inner_coverage layout qualifiers "
                             "are mutually exclusive");
         }
      }

      const bool pixel_interlock_ordered = match_layout_qualifier((yyvsp[0].identifier),
         "pixel_interlock_ordered", state) == 0;
      const bool pixel_interlock_unordered = match_layout_qualifier((yyvsp[0].identifier),
         "pixel_interlock_unordered", state) == 0;
      const bool sample_interlock_ordered = match_layout_qualifier((yyvsp[0].identifier),
         "sample_interlock_ordered", state) == 0;
      const bool sample_interlock_unordered = match_layout_qualifier((yyvsp[0].identifier),
         "sample_interlock_unordered", state) == 0;

      if (pixel_interlock_ordered + pixel_interlock_unordered +
          sample_interlock_ordered + sample_interlock_unordered > 0 &&
          state->stage != MESA_SHADER_FRAGMENT) {
         _mesa_glsl_error(& (yylsp[0]), state, "interlock layout qualifiers: "
                          "pixel_interlock_ordered, pixel_interlock_unordered, "
                          "sample_interlock_ordered and sample_interlock_unordered, "
                          "only valid in fragment shader input layout declaration.");
      } else if (pixel_interlock_ordered + pixel_interlock_unordered +
                 sample_interlock_ordered + sample_interlock_unordered > 0 &&
                 !state->ARB_fragment_shader_interlock_enable &&
                 !state->NV_fragment_shader_interlock_enable) {
         _mesa_glsl_error(& (yylsp[0]), state,
                          "interlock layout qualifier present, but the "
                          "GL_ARB_fragment_shader_interlock or "
                          "GL_NV_fragment_shader_interlock extension is not "
                          "enabled.");
      } else {
         (yyval.type_qualifier).flags.q.pixel_interlock_ordered = pixel_interlock_ordered;
         (yyval.type_qualifier).flags.q.pixel_interlock_unordered = pixel_interlock_unordered;
         (yyval.type_qualifier).flags.q.sample_interlock_ordered = sample_interlock_ordered;
         (yyval.type_qualifier).flags.q.sample_interlock_unordered = sample_interlock_unordered;
      }

      /* Layout qualifiers for tessellation evaluation shaders. */
      if (!(yyval.type_qualifier).flags.i) {
         static const struct {
            const char *s;
            GLenum e;
         } map[] = {
                 /* triangles already parsed by gs-specific code */
                 { "quads", GL_QUADS },
                 { "isolines", GL_ISOLINES },
         };
         for (unsigned i = 0; i < ARRAY_SIZE(map); i++) {
            if (match_layout_qualifier((yyvsp[0].identifier), map[i].s, state) == 0) {
               (yyval.type_qualifier).flags.q.prim_type = 1;
               (yyval.type_qualifier).prim_type = map[i].e;
               break;
            }
         }

         if ((yyval.type_qualifier).flags.i && !state->has_tessellation_shader()) {
            _mesa_glsl_error(& (yylsp[0]), state,
                             "primitive mode qualifier `%s' requires "
                             "GLSL 4.00 or ARB_tessellation_shader", (yyvsp[0].identifier));
         }
      }
      if (!(yyval.type_qualifier).flags.i) {
         static const struct {
            const char *s;
            enum gl_tess_spacing e;
         } map[] = {
                 { "equal_spacing", TESS_SPACING_EQUAL },
                 { "fractional_odd_spacing", TESS_SPACING_FRACTIONAL_ODD },
                 { "fractional_even_spacing", TESS_SPACING_FRACTIONAL_EVEN },
         };
         for (unsigned i = 0; i < ARRAY_SIZE(map); i++) {
            if (match_layout_qualifier((yyvsp[0].identifier), map[i].s, state) == 0) {
               (yyval.type_qualifier).flags.q.vertex_spacing = 1;
               (yyval.type_qualifier).vertex_spacing = map[i].e;
               break;
            }
         }

         if ((yyval.type_qualifier).flags.i && !state->has_tessellation_shader()) {
            _mesa_glsl_error(& (yylsp[0]), state,
                             "vertex spacing qualifier `%s' requires "
                             "GLSL 4.00 or ARB_tessellation_shader", (yyvsp[0].identifier));
         }
      }
      if (!(yyval.type_qualifier).flags.i) {
         if (match_layout_qualifier((yyvsp[0].identifier), "cw", state) == 0) {
            (yyval.type_qualifier).flags.q.ordering = 1;
            (yyval.type_qualifier).ordering = GL_CW;
         } else if (match_layout_qualifier((yyvsp[0].identifier), "ccw", state) == 0) {
            (yyval.type_qualifier).flags.q.ordering = 1;
            (yyval.type_qualifier).ordering = GL_CCW;
         }

         if ((yyval.type_qualifier).flags.i && !state->has_tessellation_shader()) {
            _mesa_glsl_error(& (yylsp[0]), state,
                             "ordering qualifier `%s' requires "
                             "GLSL 4.00 or ARB_tessellation_shader", (yyvsp[0].identifier));
         }
      }
      if (!(yyval.type_qualifier).flags.i) {
         if (match_layout_qualifier((yyvsp[0].identifier), "point_mode", state) == 0) {
            (yyval.type_qualifier).flags.q.point_mode = 1;
            (yyval.type_qualifier).point_mode = true;
         }

         if ((yyval.type_qualifier).flags.i && !state->has_tessellation_shader()) {
            _mesa_glsl_error(& (yylsp[0]), state,
                             "qualifier `point_mode' requires "
                             "GLSL 4.00 or ARB_tessellation_shader");
         }
      }

      if (!(yyval.type_qualifier).flags.i) {
         static const struct {
            const char *s;
            uint32_t mask;
         } map[] = {
                 { "blend_support_multiply",       BLEND_MULTIPLY },
                 { "blend_support_screen",         BLEND_SCREEN },
                 { "blend_support_overlay",        BLEND_OVERLAY },
                 { "blend_support_darken",         BLEND_DARKEN },
                 { "blend_support_lighten",        BLEND_LIGHTEN },
                 { "blend_support_colordodge",     BLEND_COLORDODGE },
                 { "blend_support_colorburn",      BLEND_COLORBURN },
                 { "blend_support_hardlight",      BLEND_HARDLIGHT },
                 { "blend_support_softlight",      BLEND_SOFTLIGHT },
                 { "blend_support_difference",     BLEND_DIFFERENCE },
                 { "blend_support_exclusion",      BLEND_EXCLUSION },
                 { "blend_support_hsl_hue",        BLEND_HSL_HUE },
                 { "blend_support_hsl_saturation", BLEND_HSL_SATURATION },
                 { "blend_support_hsl_color",      BLEND_HSL_COLOR },
                 { "blend_support_hsl_luminosity", BLEND_HSL_LUMINOSITY },
                 { "blend_support_all_equations",  BLEND_ALL },
         };
         for (unsigned i = 0; i < ARRAY_SIZE(map); i++) {
            if (match_layout_qualifier((yyvsp[0].identifier), map[i].s, state) == 0) {
               (yyval.type_qualifier).flags.q.blend_support = 1;
               state->fs_blend_support |= map[i].mask;
               break;
            }
         }

         if ((yyval.type_qualifier).flags.i &&
             !state->KHR_blend_equation_advanced_enable &&
             !state->is_version(0, 320)) {
            _mesa_glsl_error(& (yylsp[0]), state,
                             "advanced blending layout qualifiers require "
                             "ESSL 3.20 or KHR_blend_equation_advanced");
         }

         if ((yyval.type_qualifier).flags.i && state->stage != MESA_SHADER_FRAGMENT) {
            _mesa_glsl_error(& (yylsp[0]), state,
                             "advanced blending layout qualifiers only "
                             "valid in fragment shaders");
         }
      }

      /* Layout qualifiers for ARB_compute_variable_group_size. */
      if (!(yyval.type_qualifier).flags.i) {
         if (match_layout_qualifier((yyvsp[0].identifier), "local_size_variable", state) == 0) {
            (yyval.type_qualifier).flags.q.local_size_variable = 1;
         }

         if ((yyval.type_qualifier).flags.i && !state->ARB_compute_variable_group_size_enable) {
            _mesa_glsl_error(& (yylsp[0]), state,
                             "qualifier `local_size_variable` requires "
                             "ARB_compute_variable_group_size");
         }
      }

      /* Layout qualifiers for ARB_bindless_texture. */
      if (!(yyval.type_qualifier).flags.i) {
         if (match_layout_qualifier((yyvsp[0].identifier), "bindless_sampler", state) == 0)
            (yyval.type_qualifier).flags.q.bindless_sampler = 1;
         if (match_layout_qualifier((yyvsp[0].identifier), "bound_sampler", state) == 0)
            (yyval.type_qualifier).flags.q.bound_sampler = 1;

         if (state->has_shader_image_load_store()) {
            if (match_layout_qualifier((yyvsp[0].identifier), "bindless_image", state) == 0)
               (yyval.type_qualifier).flags.q.bindless_image = 1;
            if (match_layout_qualifier((yyvsp[0].identifier), "bound_image", state) == 0)
               (yyval.type_qualifier).flags.q.bound_image = 1;
         }

         if ((yyval.type_qualifier).flags.i && !state->has_bindless()) {
            _mesa_glsl_error(& (yylsp[0]), state,
                             "qualifier `%s` requires "
                             "ARB_bindless_texture", (yyvsp[0].identifier));
         }
      }

      if (!(yyval.type_qualifier).flags.i &&
          state->EXT_shader_framebuffer_fetch_non_coherent_enable) {
         if (match_layout_qualifier((yyvsp[0].identifier), "noncoherent", state) == 0)
            (yyval.type_qualifier).flags.q.non_coherent = 1;
      }

      if (!(yyval.type_qualifier).flags.i) {
         _mesa_glsl_error(& (yylsp[0]), state, "unrecognized layout identifier "
                          "`%s'", (yyvsp[0].identifier));
         YYERROR;
      }
   }
#line 3968 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 149:
#line 1651 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      memset(& (yyval.type_qualifier), 0, sizeof((yyval.type_qualifier)));
      void *ctx = state->linalloc;

      if ((yyvsp[0].expression)->oper != ast_int_constant &&
          (yyvsp[0].expression)->oper != ast_uint_constant &&
          !state->has_enhanced_layouts()) {
         _mesa_glsl_error(& (yylsp[-2]), state,
                          "compile-time constant expressions require "
                          "GLSL 4.40 or ARB_enhanced_layouts");
      }

      if (match_layout_qualifier("align", (yyvsp[-2].identifier), state) == 0) {
         if (!state->has_enhanced_layouts()) {
            _mesa_glsl_error(& (yylsp[-2]), state,
                             "align qualifier requires "
                             "GLSL 4.40 or ARB_enhanced_layouts");
         } else {
            (yyval.type_qualifier).flags.q.explicit_align = 1;
            (yyval.type_qualifier).align = (yyvsp[0].expression);
         }
      }

      if (match_layout_qualifier("location", (yyvsp[-2].identifier), state) == 0) {
         (yyval.type_qualifier).flags.q.explicit_location = 1;

         if ((yyval.type_qualifier).flags.q.attribute == 1 &&
             state->ARB_explicit_attrib_location_warn) {
            _mesa_glsl_warning(& (yylsp[-2]), state,
                               "GL_ARB_explicit_attrib_location layout "
                               "identifier `%s' used", (yyvsp[-2].identifier));
         }
         (yyval.type_qualifier).location = (yyvsp[0].expression);
      }

      if (match_layout_qualifier("component", (yyvsp[-2].identifier), state) == 0) {
         if (!state->has_enhanced_layouts()) {
            _mesa_glsl_error(& (yylsp[-2]), state,
                             "component qualifier requires "
                             "GLSL 4.40 or ARB_enhanced_layouts");
         } else {
            (yyval.type_qualifier).flags.q.explicit_component = 1;
            (yyval.type_qualifier).component = (yyvsp[0].expression);
         }
      }

      if (match_layout_qualifier("index", (yyvsp[-2].identifier), state) == 0) {
         if (state->es_shader && !state->EXT_blend_func_extended_enable) {
            _mesa_glsl_error(& (yylsp[0]), state, "index layout qualifier requires EXT_blend_func_extended");
            YYERROR;
         }

         (yyval.type_qualifier).flags.q.explicit_index = 1;
         (yyval.type_qualifier).index = (yyvsp[0].expression);
      }

      if ((state->has_420pack_or_es31() ||
           state->has_atomic_counters() ||
           state->has_shader_storage_buffer_objects()) &&
          match_layout_qualifier("binding", (yyvsp[-2].identifier), state) == 0) {
         (yyval.type_qualifier).flags.q.explicit_binding = 1;
         (yyval.type_qualifier).binding = (yyvsp[0].expression);
      }

      if ((state->has_atomic_counters() ||
           state->has_enhanced_layouts()) &&
          match_layout_qualifier("offset", (yyvsp[-2].identifier), state) == 0) {
         (yyval.type_qualifier).flags.q.explicit_offset = 1;
         (yyval.type_qualifier).offset = (yyvsp[0].expression);
      }

      if (match_layout_qualifier("max_vertices", (yyvsp[-2].identifier), state) == 0) {
         (yyval.type_qualifier).flags.q.max_vertices = 1;
         (yyval.type_qualifier).max_vertices = new(ctx) ast_layout_expression((yylsp[-2]), (yyvsp[0].expression));
         if (!state->has_geometry_shader()) {
            _mesa_glsl_error(& (yylsp[0]), state,
                             "#version 150 max_vertices qualifier "
                             "specified", (yyvsp[0].expression));
         }
      }

      if (state->stage == MESA_SHADER_GEOMETRY) {
         if (match_layout_qualifier("stream", (yyvsp[-2].identifier), state) == 0 &&
             state->check_explicit_attrib_stream_allowed(& (yylsp[0]))) {
            (yyval.type_qualifier).flags.q.stream = 1;
            (yyval.type_qualifier).flags.q.explicit_stream = 1;
            (yyval.type_qualifier).stream = (yyvsp[0].expression);
         }
      }

      if (state->has_enhanced_layouts()) {
         if (match_layout_qualifier("xfb_buffer", (yyvsp[-2].identifier), state) == 0) {
            (yyval.type_qualifier).flags.q.xfb_buffer = 1;
            (yyval.type_qualifier).flags.q.explicit_xfb_buffer = 1;
            (yyval.type_qualifier).xfb_buffer = (yyvsp[0].expression);
         }

         if (match_layout_qualifier("xfb_offset", (yyvsp[-2].identifier), state) == 0) {
            (yyval.type_qualifier).flags.q.explicit_xfb_offset = 1;
            (yyval.type_qualifier).offset = (yyvsp[0].expression);
         }

         if (match_layout_qualifier("xfb_stride", (yyvsp[-2].identifier), state) == 0) {
            (yyval.type_qualifier).flags.q.xfb_stride = 1;
            (yyval.type_qualifier).flags.q.explicit_xfb_stride = 1;
            (yyval.type_qualifier).xfb_stride = (yyvsp[0].expression);
         }
      }

      static const char * const local_size_qualifiers[3] = {
         "local_size_x",
         "local_size_y",
         "local_size_z",
      };
      for (int i = 0; i < 3; i++) {
         if (match_layout_qualifier(local_size_qualifiers[i], (yyvsp[-2].identifier),
                                    state) == 0) {
            if (!state->has_compute_shader()) {
               _mesa_glsl_error(& (yylsp[0]), state,
                                "%s qualifier requires GLSL 4.30 or "
                                "GLSL ES 3.10 or ARB_compute_shader",
                                local_size_qualifiers[i]);
               YYERROR;
            } else {
               (yyval.type_qualifier).flags.q.local_size |= (1 << i);
               (yyval.type_qualifier).local_size[i] = new(ctx) ast_layout_expression((yylsp[-2]), (yyvsp[0].expression));
            }
            break;
         }
      }

      if (match_layout_qualifier("invocations", (yyvsp[-2].identifier), state) == 0) {
         (yyval.type_qualifier).flags.q.invocations = 1;
         (yyval.type_qualifier).invocations = new(ctx) ast_layout_expression((yylsp[-2]), (yyvsp[0].expression));
         if (!state->is_version(400, 320) &&
             !state->ARB_gpu_shader5_enable &&
             !state->OES_geometry_shader_enable &&
             !state->EXT_geometry_shader_enable) {
            _mesa_glsl_error(& (yylsp[0]), state,
                             "GL_ARB_gpu_shader5 invocations "
                             "qualifier specified", (yyvsp[0].expression));
         }
      }

      /* Layout qualifiers for tessellation control shaders. */
      if (match_layout_qualifier("vertices", (yyvsp[-2].identifier), state) == 0) {
         (yyval.type_qualifier).flags.q.vertices = 1;
         (yyval.type_qualifier).vertices = new(ctx) ast_layout_expression((yylsp[-2]), (yyvsp[0].expression));
         if (!state->has_tessellation_shader()) {
            _mesa_glsl_error(& (yylsp[-2]), state,
                             "vertices qualifier requires GLSL 4.00 or "
                             "ARB_tessellation_shader");
         }
      }

      /* If the identifier didn't match any known layout identifiers,
       * emit an error.
       */
      if (!(yyval.type_qualifier).flags.i) {
         _mesa_glsl_error(& (yylsp[-2]), state, "unrecognized layout identifier "
                          "`%s'", (yyvsp[-2].identifier));
         YYERROR;
      }
   }
#line 4137 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 150:
#line 1816 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      (yyval.type_qualifier) = (yyvsp[0].type_qualifier);
      /* Layout qualifiers for ARB_uniform_buffer_object. */
      if ((yyval.type_qualifier).flags.q.uniform && !state->has_uniform_buffer_objects()) {
         _mesa_glsl_error(& (yylsp[0]), state,
                          "#version 140 / GL_ARB_uniform_buffer_object "
                          "layout qualifier `%s' is used", (yyvsp[0].type_qualifier));
      } else if ((yyval.type_qualifier).flags.q.uniform && state->ARB_uniform_buffer_object_warn) {
         _mesa_glsl_warning(& (yylsp[0]), state,
                            "#version 140 / GL_ARB_uniform_buffer_object "
                            "layout qualifier `%s' is used", (yyvsp[0].type_qualifier));
      }
   }
#line 4155 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 151:
#line 1842 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      memset(& (yyval.type_qualifier), 0, sizeof((yyval.type_qualifier)));
      (yyval.type_qualifier).flags.q.row_major = 1;
   }
#line 4164 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 152:
#line 1847 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      memset(& (yyval.type_qualifier), 0, sizeof((yyval.type_qualifier)));
      (yyval.type_qualifier).flags.q.packed = 1;
   }
#line 4173 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 153:
#line 1852 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      memset(& (yyval.type_qualifier), 0, sizeof((yyval.type_qualifier)));
      (yyval.type_qualifier).flags.q.shared = 1;
   }
#line 4182 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 154:
#line 1860 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      memset(& (yyval.type_qualifier), 0, sizeof((yyval.type_qualifier)));
      (yyval.type_qualifier).flags.q.subroutine = 1;
   }
#line 4191 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 155:
#line 1865 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      memset(& (yyval.type_qualifier), 0, sizeof((yyval.type_qualifier)));
      (yyval.type_qualifier).flags.q.subroutine = 1;
      (yyval.type_qualifier).subroutine_list = (yyvsp[-1].subroutine_list);
   }
#line 4201 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 156:
#line 1874 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
        void *ctx = state->linalloc;
        ast_declaration *decl = new(ctx)  ast_declaration((yyvsp[0].identifier), NULL, NULL);
        decl->set_location((yylsp[0]));

        (yyval.subroutine_list) = new(ctx) ast_subroutine_list();
        (yyval.subroutine_list)->declarations.push_tail(&decl->link);
   }
#line 4214 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 157:
#line 1883 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
        void *ctx = state->linalloc;
        ast_declaration *decl = new(ctx)  ast_declaration((yyvsp[0].identifier), NULL, NULL);
        decl->set_location((yylsp[0]));

        (yyval.subroutine_list) = (yyvsp[-2].subroutine_list);
        (yyval.subroutine_list)->declarations.push_tail(&decl->link);
   }
#line 4227 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 158:
#line 1895 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      memset(& (yyval.type_qualifier), 0, sizeof((yyval.type_qualifier)));
      (yyval.type_qualifier).flags.q.smooth = 1;
   }
#line 4236 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 159:
#line 1900 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      memset(& (yyval.type_qualifier), 0, sizeof((yyval.type_qualifier)));
      (yyval.type_qualifier).flags.q.flat = 1;
   }
#line 4245 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 160:
#line 1905 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      memset(& (yyval.type_qualifier), 0, sizeof((yyval.type_qualifier)));
      (yyval.type_qualifier).flags.q.noperspective = 1;
   }
#line 4254 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 161:
#line 1914 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      memset(& (yyval.type_qualifier), 0, sizeof((yyval.type_qualifier)));
      (yyval.type_qualifier).flags.q.invariant = 1;
   }
#line 4263 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 162:
#line 1919 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      memset(& (yyval.type_qualifier), 0, sizeof((yyval.type_qualifier)));
      (yyval.type_qualifier).flags.q.precise = 1;
   }
#line 4272 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 169:
#line 1930 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      memset(&(yyval.type_qualifier), 0, sizeof((yyval.type_qualifier)));
      (yyval.type_qualifier).precision = (yyvsp[0].n);
   }
#line 4281 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 170:
#line 1948 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      if ((yyvsp[0].type_qualifier).flags.q.precise)
         _mesa_glsl_error(&(yylsp[-1]), state, "duplicate \"precise\" qualifier");

      (yyval.type_qualifier) = (yyvsp[0].type_qualifier);
      (yyval.type_qualifier).flags.q.precise = 1;
   }
#line 4293 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 171:
#line 1956 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      if ((yyvsp[0].type_qualifier).flags.q.invariant)
         _mesa_glsl_error(&(yylsp[-1]), state, "duplicate \"invariant\" qualifier");

      if (!state->has_420pack_or_es31() && (yyvsp[0].type_qualifier).flags.q.precise)
         _mesa_glsl_error(&(yylsp[-1]), state,
                          "\"invariant\" must come after \"precise\"");

      (yyval.type_qualifier) = (yyvsp[0].type_qualifier);
      (yyval.type_qualifier).flags.q.invariant = 1;

      /* GLSL ES 3.00 spec, section 4.6.1 "The Invariant Qualifier":
       *
       * "Only variables output from a shader can be candidates for invariance.
       * This includes user-defined output variables and the built-in output
       * variables. As only outputs can be declared as invariant, an invariant
       * output from one shader stage will still match an input of a subsequent
       * stage without the input being declared as invariant."
       *
       * On the desktop side, this text first appears in GLSL 4.30.
       */
      if (state->is_version(430, 300) && (yyval.type_qualifier).flags.q.in)
         _mesa_glsl_error(&(yylsp[-1]), state, "invariant qualifiers cannot be used with shader inputs");
   }
#line 4322 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 172:
#line 1981 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      /* Section 4.3 of the GLSL 1.40 specification states:
       * "...qualified with one of these interpolation qualifiers"
       *
       * GLSL 1.30 claims to allow "one or more", but insists that:
       * "These interpolation qualifiers may only precede the qualifiers in,
       *  centroid in, out, or centroid out in a declaration."
       *
       * ...which means that e.g. smooth can't precede smooth, so there can be
       * only one after all, and the 1.40 text is a clarification, not a change.
       */
      if ((yyvsp[0].type_qualifier).has_interpolation())
         _mesa_glsl_error(&(yylsp[-1]), state, "duplicate interpolation qualifier");

      if (!state->has_420pack_or_es31() &&
          ((yyvsp[0].type_qualifier).flags.q.precise || (yyvsp[0].type_qualifier).flags.q.invariant)) {
         _mesa_glsl_error(&(yylsp[-1]), state, "interpolation qualifiers must come "
                          "after \"precise\" or \"invariant\"");
      }

      (yyval.type_qualifier) = (yyvsp[-1].type_qualifier);
      (yyval.type_qualifier).merge_qualifier(&(yylsp[-1]), state, (yyvsp[0].type_qualifier), false);
   }
#line 4350 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 173:
#line 2005 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      /* In the absence of ARB_shading_language_420pack, layout qualifiers may
       * appear no later than auxiliary storage qualifiers. There is no
       * particularly clear spec language mandating this, but in all examples
       * the layout qualifier precedes the storage qualifier.
       *
       * We allow combinations of layout with interpolation, invariant or
       * precise qualifiers since these are useful in ARB_separate_shader_objects.
       * There is no clear spec guidance on this either.
       */
      (yyval.type_qualifier) = (yyvsp[-1].type_qualifier);
      (yyval.type_qualifier).merge_qualifier(& (yylsp[-1]), state, (yyvsp[0].type_qualifier), false, (yyvsp[0].type_qualifier).has_layout());
   }
#line 4368 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 174:
#line 2019 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      (yyval.type_qualifier) = (yyvsp[-1].type_qualifier);
      (yyval.type_qualifier).merge_qualifier(&(yylsp[-1]), state, (yyvsp[0].type_qualifier), false);
   }
#line 4377 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 175:
#line 2024 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      if ((yyvsp[0].type_qualifier).has_auxiliary_storage()) {
         _mesa_glsl_error(&(yylsp[-1]), state,
                          "duplicate auxiliary storage qualifier (centroid or sample)");
      }

      if (!state->has_420pack_or_es31() &&
          ((yyvsp[0].type_qualifier).flags.q.precise || (yyvsp[0].type_qualifier).flags.q.invariant ||
           (yyvsp[0].type_qualifier).has_interpolation() || (yyvsp[0].type_qualifier).has_layout())) {
         _mesa_glsl_error(&(yylsp[-1]), state, "auxiliary storage qualifiers must come "
                          "just before storage qualifiers");
      }
      (yyval.type_qualifier) = (yyvsp[-1].type_qualifier);
      (yyval.type_qualifier).merge_qualifier(&(yylsp[-1]), state, (yyvsp[0].type_qualifier), false);
   }
#line 4397 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 176:
#line 2040 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      /* Section 4.3 of the GLSL 1.20 specification states:
       * "Variable declarations may have a storage qualifier specified..."
       *  1.30 clarifies this to "may have one storage qualifier".
       */
      if ((yyvsp[0].type_qualifier).has_storage())
         _mesa_glsl_error(&(yylsp[-1]), state, "duplicate storage qualifier");

      if (!state->has_420pack_or_es31() &&
          ((yyvsp[0].type_qualifier).flags.q.precise || (yyvsp[0].type_qualifier).flags.q.invariant || (yyvsp[0].type_qualifier).has_interpolation() ||
           (yyvsp[0].type_qualifier).has_layout() || (yyvsp[0].type_qualifier).has_auxiliary_storage())) {
         _mesa_glsl_error(&(yylsp[-1]), state, "storage qualifiers must come after "
                          "precise, invariant, interpolation, layout and auxiliary "
                          "storage qualifiers");
      }

      (yyval.type_qualifier) = (yyvsp[-1].type_qualifier);
      (yyval.type_qualifier).merge_qualifier(&(yylsp[-1]), state, (yyvsp[0].type_qualifier), false);
   }
#line 4421 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 177:
#line 2060 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      if ((yyvsp[0].type_qualifier).precision != ast_precision_none)
         _mesa_glsl_error(&(yylsp[-1]), state, "duplicate precision qualifier");

      if (!(state->has_420pack_or_es31()) &&
          (yyvsp[0].type_qualifier).flags.i != 0)
         _mesa_glsl_error(&(yylsp[-1]), state, "precision qualifiers must come last");

      (yyval.type_qualifier) = (yyvsp[0].type_qualifier);
      (yyval.type_qualifier).precision = (yyvsp[-1].n);
   }
#line 4437 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 178:
#line 2072 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      (yyval.type_qualifier) = (yyvsp[-1].type_qualifier);
      (yyval.type_qualifier).merge_qualifier(&(yylsp[-1]), state, (yyvsp[0].type_qualifier), false);
   }
#line 4446 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 179:
#line 2080 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      memset(& (yyval.type_qualifier), 0, sizeof((yyval.type_qualifier)));
      (yyval.type_qualifier).flags.q.centroid = 1;
   }
#line 4455 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 180:
#line 2085 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      memset(& (yyval.type_qualifier), 0, sizeof((yyval.type_qualifier)));
      (yyval.type_qualifier).flags.q.sample = 1;
   }
#line 4464 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 181:
#line 2090 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      memset(& (yyval.type_qualifier), 0, sizeof((yyval.type_qualifier)));
      (yyval.type_qualifier).flags.q.patch = 1;
   }
#line 4473 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 182:
#line 2097 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      memset(& (yyval.type_qualifier), 0, sizeof((yyval.type_qualifier)));
      (yyval.type_qualifier).flags.q.constant = 1;
   }
#line 4482 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 183:
#line 2102 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      memset(& (yyval.type_qualifier), 0, sizeof((yyval.type_qualifier)));
      (yyval.type_qualifier).flags.q.attribute = 1;
   }
#line 4491 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 184:
#line 2107 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      memset(& (yyval.type_qualifier), 0, sizeof((yyval.type_qualifier)));
      (yyval.type_qualifier).flags.q.varying = 1;
   }
#line 4500 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 185:
#line 2112 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      memset(& (yyval.type_qualifier), 0, sizeof((yyval.type_qualifier)));
      (yyval.type_qualifier).flags.q.in = 1;
   }
#line 4509 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 186:
#line 2117 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      memset(& (yyval.type_qualifier), 0, sizeof((yyval.type_qualifier)));
      (yyval.type_qualifier).flags.q.out = 1;

      if (state->stage == MESA_SHADER_GEOMETRY &&
          state->has_explicit_attrib_stream()) {
         /* Section 4.3.8.2 (Output Layout Qualifiers) of the GLSL 4.00
          * spec says:
          *
          *     "If the block or variable is declared with the stream
          *     identifier, it is associated with the specified stream;
          *     otherwise, it is associated with the current default stream."
          */
          (yyval.type_qualifier).flags.q.stream = 1;
          (yyval.type_qualifier).flags.q.explicit_stream = 0;
          (yyval.type_qualifier).stream = state->out_qualifier->stream;
      }

      if (state->has_enhanced_layouts()) {
          (yyval.type_qualifier).flags.q.xfb_buffer = 1;
          (yyval.type_qualifier).flags.q.explicit_xfb_buffer = 0;
          (yyval.type_qualifier).xfb_buffer = state->out_qualifier->xfb_buffer;
      }
   }
#line 4538 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 187:
#line 2142 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      memset(& (yyval.type_qualifier), 0, sizeof((yyval.type_qualifier)));
      (yyval.type_qualifier).flags.q.in = 1;
      (yyval.type_qualifier).flags.q.out = 1;

      if (!state->has_framebuffer_fetch() ||
          !state->is_version(130, 300) ||
          state->stage != MESA_SHADER_FRAGMENT)
         _mesa_glsl_error(&(yylsp[0]), state, "A single interface variable cannot be "
                          "declared as both input and output");
   }
#line 4554 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 188:
#line 2154 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      memset(& (yyval.type_qualifier), 0, sizeof((yyval.type_qualifier)));
      (yyval.type_qualifier).flags.q.uniform = 1;
   }
#line 4563 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 189:
#line 2159 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      memset(& (yyval.type_qualifier), 0, sizeof((yyval.type_qualifier)));
      (yyval.type_qualifier).flags.q.buffer = 1;
   }
#line 4572 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 190:
#line 2164 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      memset(& (yyval.type_qualifier), 0, sizeof((yyval.type_qualifier)));
      (yyval.type_qualifier).flags.q.shared_storage = 1;
   }
#line 4581 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 191:
#line 2172 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      memset(& (yyval.type_qualifier), 0, sizeof((yyval.type_qualifier)));
      (yyval.type_qualifier).flags.q.coherent = 1;
   }
#line 4590 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 192:
#line 2177 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      memset(& (yyval.type_qualifier), 0, sizeof((yyval.type_qualifier)));
      (yyval.type_qualifier).flags.q._volatile = 1;
   }
#line 4599 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 193:
#line 2182 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      STATIC_ASSERT(sizeof((yyval.type_qualifier).flags.q) <= sizeof((yyval.type_qualifier).flags.i));
      memset(& (yyval.type_qualifier), 0, sizeof((yyval.type_qualifier)));
      (yyval.type_qualifier).flags.q.restrict_flag = 1;
   }
#line 4609 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 194:
#line 2188 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      memset(& (yyval.type_qualifier), 0, sizeof((yyval.type_qualifier)));
      (yyval.type_qualifier).flags.q.read_only = 1;
   }
#line 4618 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 195:
#line 2193 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      memset(& (yyval.type_qualifier), 0, sizeof((yyval.type_qualifier)));
      (yyval.type_qualifier).flags.q.write_only = 1;
   }
#line 4627 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 196:
#line 2201 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      void *ctx = state->linalloc;
      (yyval.array_specifier) = new(ctx) ast_array_specifier((yylsp[-1]), new(ctx) ast_expression(
                                                  ast_unsized_array_dim, NULL,
                                                  NULL, NULL));
      (yyval.array_specifier)->set_location_range((yylsp[-1]), (yylsp[0]));
   }
#line 4639 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 197:
#line 2209 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      void *ctx = state->linalloc;
      (yyval.array_specifier) = new(ctx) ast_array_specifier((yylsp[-2]), (yyvsp[-1].expression));
      (yyval.array_specifier)->set_location_range((yylsp[-2]), (yylsp[0]));
   }
#line 4649 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 198:
#line 2215 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      void *ctx = state->linalloc;
      (yyval.array_specifier) = (yyvsp[-2].array_specifier);

      if (state->check_arrays_of_arrays_allowed(& (yylsp[-2]))) {
         (yyval.array_specifier)->add_dimension(new(ctx) ast_expression(ast_unsized_array_dim, NULL,
                                                   NULL, NULL));
      }
   }
#line 4663 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 199:
#line 2225 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      (yyval.array_specifier) = (yyvsp[-3].array_specifier);

      if (state->check_arrays_of_arrays_allowed(& (yylsp[-3]))) {
         (yyval.array_specifier)->add_dimension((yyvsp[-1].expression));
      }
   }
#line 4675 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 201:
#line 2237 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      (yyval.type_specifier) = (yyvsp[-1].type_specifier);
      (yyval.type_specifier)->array_specifier = (yyvsp[0].array_specifier);
   }
#line 4684 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 202:
#line 2245 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      void *ctx = state->linalloc;
      (yyval.type_specifier) = new(ctx) ast_type_specifier((yyvsp[0].type));
      (yyval.type_specifier)->set_location((yylsp[0]));
   }
#line 4694 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 203:
#line 2251 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      void *ctx = state->linalloc;
      (yyval.type_specifier) = new(ctx) ast_type_specifier((yyvsp[0].struct_specifier));
      (yyval.type_specifier)->set_location((yylsp[0]));
   }
#line 4704 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 204:
#line 2257 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      void *ctx = state->linalloc;
      (yyval.type_specifier) = new(ctx) ast_type_specifier((yyvsp[0].identifier));
      (yyval.type_specifier)->set_location((yylsp[0]));
   }
#line 4714 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 205:
#line 2265 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    { (yyval.type) = glsl_type::void_type; }
#line 4720 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 206:
#line 2266 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    { (yyval.type) = (yyvsp[0].type); }
#line 4726 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 207:
#line 2271 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      state->check_precision_qualifiers_allowed(&(yylsp[0]));
      (yyval.n) = ast_precision_high;
   }
#line 4735 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 208:
#line 2276 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      state->check_precision_qualifiers_allowed(&(yylsp[0]));
      (yyval.n) = ast_precision_medium;
   }
#line 4744 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 209:
#line 2281 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      state->check_precision_qualifiers_allowed(&(yylsp[0]));
      (yyval.n) = ast_precision_low;
   }
#line 4753 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 210:
#line 2289 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      void *ctx = state->linalloc;
      (yyval.struct_specifier) = new(ctx) ast_struct_specifier((yyvsp[-3].identifier), (yyvsp[-1].declarator_list));
      (yyval.struct_specifier)->set_location_range((yylsp[-3]), (yylsp[0]));
      state->symbols->add_type((yyvsp[-3].identifier), glsl_type::void_type);
   }
#line 4764 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 211:
#line 2296 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      void *ctx = state->linalloc;

      /* All anonymous structs have the same name. This simplifies matching of
       * globals whose type is an unnamed struct.
       *
       * It also avoids a memory leak when the same shader is compiled over and
       * over again.
       */
      (yyval.struct_specifier) = new(ctx) ast_struct_specifier("#anon_struct", (yyvsp[-1].declarator_list));

      (yyval.struct_specifier)->set_location_range((yylsp[-2]), (yylsp[0]));
   }
#line 4782 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 212:
#line 2313 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      (yyval.declarator_list) = (yyvsp[0].declarator_list);
      (yyvsp[0].declarator_list)->link.self_link();
   }
#line 4791 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 213:
#line 2318 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      (yyval.declarator_list) = (yyvsp[-1].declarator_list);
      (yyval.declarator_list)->link.insert_before(& (yyvsp[0].declarator_list)->link);
   }
#line 4800 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 214:
#line 2326 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      void *ctx = state->linalloc;
      ast_fully_specified_type *const type = (yyvsp[-2].fully_specified_type);
      type->set_location((yylsp[-2]));

      if (state->has_bindless()) {
         ast_type_qualifier input_layout_mask;

         /* Allow to declare qualifiers for images. */
         input_layout_mask.flags.i = 0;
         input_layout_mask.flags.q.coherent = 1;
         input_layout_mask.flags.q._volatile = 1;
         input_layout_mask.flags.q.restrict_flag = 1;
         input_layout_mask.flags.q.read_only = 1;
         input_layout_mask.flags.q.write_only = 1;
         input_layout_mask.flags.q.explicit_image_format = 1;

         if ((type->qualifier.flags.i & ~input_layout_mask.flags.i) != 0) {
            _mesa_glsl_error(&(yylsp[-2]), state,
                             "only precision and image qualifiers may be "
                             "applied to structure members");
         }
      } else {
         if (type->qualifier.flags.i != 0)
            _mesa_glsl_error(&(yylsp[-2]), state,
                             "only precision qualifiers may be applied to "
                             "structure members");
      }

      (yyval.declarator_list) = new(ctx) ast_declarator_list(type);
      (yyval.declarator_list)->set_location((yylsp[-1]));

      (yyval.declarator_list)->declarations.push_degenerate_list_at_head(& (yyvsp[-1].declaration)->link);
   }
#line 4839 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 215:
#line 2364 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      (yyval.declaration) = (yyvsp[0].declaration);
      (yyvsp[0].declaration)->link.self_link();
   }
#line 4848 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 216:
#line 2369 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      (yyval.declaration) = (yyvsp[-2].declaration);
      (yyval.declaration)->link.insert_before(& (yyvsp[0].declaration)->link);
   }
#line 4857 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 217:
#line 2377 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      void *ctx = state->linalloc;
      (yyval.declaration) = new(ctx) ast_declaration((yyvsp[0].identifier), NULL, NULL);
      (yyval.declaration)->set_location((yylsp[0]));
   }
#line 4867 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 218:
#line 2383 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      void *ctx = state->linalloc;
      (yyval.declaration) = new(ctx) ast_declaration((yyvsp[-1].identifier), (yyvsp[0].array_specifier), NULL);
      (yyval.declaration)->set_location_range((yylsp[-1]), (yylsp[0]));
   }
#line 4877 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 220:
#line 2393 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      (yyval.expression) = (yyvsp[-1].expression);
   }
#line 4885 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 221:
#line 2397 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      (yyval.expression) = (yyvsp[-2].expression);
   }
#line 4893 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 222:
#line 2404 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      void *ctx = state->linalloc;
      (yyval.expression) = new(ctx) ast_aggregate_initializer();
      (yyval.expression)->set_location((yylsp[0]));
      (yyval.expression)->expressions.push_tail(& (yyvsp[0].expression)->link);
   }
#line 4904 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 223:
#line 2411 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      (yyvsp[-2].expression)->expressions.push_tail(& (yyvsp[0].expression)->link);
   }
#line 4912 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 225:
#line 2423 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    { (yyval.node) = (ast_node *) (yyvsp[0].compound_statement); }
#line 4918 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 233:
#line 2438 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      void *ctx = state->linalloc;
      (yyval.compound_statement) = new(ctx) ast_compound_statement(true, NULL);
      (yyval.compound_statement)->set_location_range((yylsp[-1]), (yylsp[0]));
   }
#line 4928 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 234:
#line 2444 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      state->symbols->push_scope();
   }
#line 4936 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 235:
#line 2448 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      void *ctx = state->linalloc;
      (yyval.compound_statement) = new(ctx) ast_compound_statement(true, (yyvsp[-1].node));
      (yyval.compound_statement)->set_location_range((yylsp[-3]), (yylsp[0]));
      state->symbols->pop_scope();
   }
#line 4947 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 236:
#line 2457 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    { (yyval.node) = (ast_node *) (yyvsp[0].compound_statement); }
#line 4953 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 238:
#line 2463 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      void *ctx = state->linalloc;
      (yyval.compound_statement) = new(ctx) ast_compound_statement(false, NULL);
      (yyval.compound_statement)->set_location_range((yylsp[-1]), (yylsp[0]));
   }
#line 4963 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 239:
#line 2469 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      void *ctx = state->linalloc;
      (yyval.compound_statement) = new(ctx) ast_compound_statement(false, (yyvsp[-1].node));
      (yyval.compound_statement)->set_location_range((yylsp[-2]), (yylsp[0]));
   }
#line 4973 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 240:
#line 2478 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      if ((yyvsp[0].node) == NULL) {
         _mesa_glsl_error(& (yylsp[0]), state, "<nil> statement");
         assert((yyvsp[0].node) != NULL);
      }

      (yyval.node) = (yyvsp[0].node);
      (yyval.node)->link.self_link();
   }
#line 4987 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 241:
#line 2488 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      if ((yyvsp[0].node) == NULL) {
         _mesa_glsl_error(& (yylsp[0]), state, "<nil> statement");
         assert((yyvsp[0].node) != NULL);
      }
      (yyval.node) = (yyvsp[-1].node);
      (yyval.node)->link.insert_before(& (yyvsp[0].node)->link);
   }
#line 5000 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 242:
#line 2500 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      void *ctx = state->linalloc;
      (yyval.node) = new(ctx) ast_expression_statement(NULL);
      (yyval.node)->set_location((yylsp[0]));
   }
#line 5010 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 243:
#line 2506 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      void *ctx = state->linalloc;
      (yyval.node) = new(ctx) ast_expression_statement((yyvsp[-1].expression));
      (yyval.node)->set_location((yylsp[-1]));
   }
#line 5020 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 244:
#line 2515 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      (yyval.node) = new(state->linalloc) ast_selection_statement((yyvsp[-2].expression), (yyvsp[0].selection_rest_statement).then_statement,
                                                        (yyvsp[0].selection_rest_statement).else_statement);
      (yyval.node)->set_location_range((yylsp[-4]), (yylsp[0]));
   }
#line 5030 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 245:
#line 2524 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      (yyval.selection_rest_statement).then_statement = (yyvsp[-2].node);
      (yyval.selection_rest_statement).else_statement = (yyvsp[0].node);
   }
#line 5039 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 246:
#line 2529 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      (yyval.selection_rest_statement).then_statement = (yyvsp[0].node);
      (yyval.selection_rest_statement).else_statement = NULL;
   }
#line 5048 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 247:
#line 2537 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      (yyval.node) = (ast_node *) (yyvsp[0].expression);
   }
#line 5056 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 248:
#line 2541 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      void *ctx = state->linalloc;
      ast_declaration *decl = new(ctx) ast_declaration((yyvsp[-2].identifier), NULL, (yyvsp[0].expression));
      ast_declarator_list *declarator = new(ctx) ast_declarator_list((yyvsp[-3].fully_specified_type));
      decl->set_location_range((yylsp[-2]), (yylsp[0]));
      declarator->set_location((yylsp[-3]));

      declarator->declarations.push_tail(&decl->link);
      (yyval.node) = declarator;
   }
#line 5071 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 249:
#line 2559 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      (yyval.node) = new(state->linalloc) ast_switch_statement((yyvsp[-2].expression), (yyvsp[0].switch_body));
      (yyval.node)->set_location_range((yylsp[-4]), (yylsp[0]));
   }
#line 5080 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 250:
#line 2567 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      (yyval.switch_body) = new(state->linalloc) ast_switch_body(NULL);
      (yyval.switch_body)->set_location_range((yylsp[-1]), (yylsp[0]));
   }
#line 5089 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 251:
#line 2572 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      (yyval.switch_body) = new(state->linalloc) ast_switch_body((yyvsp[-1].case_statement_list));
      (yyval.switch_body)->set_location_range((yylsp[-2]), (yylsp[0]));
   }
#line 5098 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 252:
#line 2580 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      (yyval.case_label) = new(state->linalloc) ast_case_label((yyvsp[-1].expression));
      (yyval.case_label)->set_location((yylsp[-1]));
   }
#line 5107 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 253:
#line 2585 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      (yyval.case_label) = new(state->linalloc) ast_case_label(NULL);
      (yyval.case_label)->set_location((yylsp[0]));
   }
#line 5116 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 254:
#line 2593 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      ast_case_label_list *labels = new(state->linalloc) ast_case_label_list();

      labels->labels.push_tail(& (yyvsp[0].case_label)->link);
      (yyval.case_label_list) = labels;
      (yyval.case_label_list)->set_location((yylsp[0]));
   }
#line 5128 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 255:
#line 2601 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      (yyval.case_label_list) = (yyvsp[-1].case_label_list);
      (yyval.case_label_list)->labels.push_tail(& (yyvsp[0].case_label)->link);
   }
#line 5137 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 256:
#line 2609 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      ast_case_statement *stmts = new(state->linalloc) ast_case_statement((yyvsp[-1].case_label_list));
      stmts->set_location((yylsp[0]));

      stmts->stmts.push_tail(& (yyvsp[0].node)->link);
      (yyval.case_statement) = stmts;
   }
#line 5149 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 257:
#line 2617 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      (yyval.case_statement) = (yyvsp[-1].case_statement);
      (yyval.case_statement)->stmts.push_tail(& (yyvsp[0].node)->link);
   }
#line 5158 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 258:
#line 2625 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      ast_case_statement_list *cases= new(state->linalloc) ast_case_statement_list();
      cases->set_location((yylsp[0]));

      cases->cases.push_tail(& (yyvsp[0].case_statement)->link);
      (yyval.case_statement_list) = cases;
   }
#line 5170 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 259:
#line 2633 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      (yyval.case_statement_list) = (yyvsp[-1].case_statement_list);
      (yyval.case_statement_list)->cases.push_tail(& (yyvsp[0].case_statement)->link);
   }
#line 5179 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 260:
#line 2641 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      void *ctx = state->linalloc;
      (yyval.node) = new(ctx) ast_iteration_statement(ast_iteration_statement::ast_while,
                                            NULL, (yyvsp[-2].node), NULL, (yyvsp[0].node));
      (yyval.node)->set_location_range((yylsp[-4]), (yylsp[-1]));
   }
#line 5190 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 261:
#line 2648 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      void *ctx = state->linalloc;
      (yyval.node) = new(ctx) ast_iteration_statement(ast_iteration_statement::ast_do_while,
                                            NULL, (yyvsp[-2].expression), NULL, (yyvsp[-5].node));
      (yyval.node)->set_location_range((yylsp[-6]), (yylsp[-1]));
   }
#line 5201 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 262:
#line 2655 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      void *ctx = state->linalloc;
      (yyval.node) = new(ctx) ast_iteration_statement(ast_iteration_statement::ast_for,
                                            (yyvsp[-3].node), (yyvsp[-2].for_rest_statement).cond, (yyvsp[-2].for_rest_statement).rest, (yyvsp[0].node));
      (yyval.node)->set_location_range((yylsp[-5]), (yylsp[0]));
   }
#line 5212 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 266:
#line 2671 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      (yyval.node) = NULL;
   }
#line 5220 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 267:
#line 2678 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      (yyval.for_rest_statement).cond = (yyvsp[-1].node);
      (yyval.for_rest_statement).rest = NULL;
   }
#line 5229 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 268:
#line 2683 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      (yyval.for_rest_statement).cond = (yyvsp[-2].node);
      (yyval.for_rest_statement).rest = (yyvsp[0].expression);
   }
#line 5238 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 269:
#line 2692 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      void *ctx = state->linalloc;
      (yyval.node) = new(ctx) ast_jump_statement(ast_jump_statement::ast_continue, NULL);
      (yyval.node)->set_location((yylsp[-1]));
   }
#line 5248 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 270:
#line 2698 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      void *ctx = state->linalloc;
      (yyval.node) = new(ctx) ast_jump_statement(ast_jump_statement::ast_break, NULL);
      (yyval.node)->set_location((yylsp[-1]));
   }
#line 5258 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 271:
#line 2704 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      void *ctx = state->linalloc;
      (yyval.node) = new(ctx) ast_jump_statement(ast_jump_statement::ast_return, NULL);
      (yyval.node)->set_location((yylsp[-1]));
   }
#line 5268 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 272:
#line 2710 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      void *ctx = state->linalloc;
      (yyval.node) = new(ctx) ast_jump_statement(ast_jump_statement::ast_return, (yyvsp[-1].expression));
      (yyval.node)->set_location_range((yylsp[-2]), (yylsp[-1]));
   }
#line 5278 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 273:
#line 2716 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      void *ctx = state->linalloc;
      (yyval.node) = new(ctx) ast_jump_statement(ast_jump_statement::ast_discard, NULL);
      (yyval.node)->set_location((yylsp[-1]));
   }
#line 5288 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 274:
#line 2724 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    { (yyval.node) = (yyvsp[0].function_definition); }
#line 5294 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 275:
#line 2725 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    { (yyval.node) = (yyvsp[0].node); }
#line 5300 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 276:
#line 2726 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    { (yyval.node) = NULL; }
#line 5306 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 277:
#line 2727 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    { (yyval.node) = (yyvsp[0].node); }
#line 5312 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 278:
#line 2728 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    { (yyval.node) = NULL; }
#line 5318 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 279:
#line 2733 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      void *ctx = state->linalloc;
      (yyval.function_definition) = new(ctx) ast_function_definition();
      (yyval.function_definition)->set_location_range((yylsp[-1]), (yylsp[0]));
      (yyval.function_definition)->prototype = (yyvsp[-1].function);
      (yyval.function_definition)->body = (yyvsp[0].compound_statement);

      state->symbols->pop_scope();
   }
#line 5332 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 280:
#line 2747 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      (yyval.node) = (yyvsp[0].interface_block);
   }
#line 5340 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 281:
#line 2751 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      ast_interface_block *block = (ast_interface_block *) (yyvsp[0].node);

      if (!(yyvsp[-1].type_qualifier).merge_qualifier(& (yylsp[-1]), state, block->layout, false,
                              block->layout.has_layout())) {
         YYERROR;
      }

      block->layout = (yyvsp[-1].type_qualifier);

      (yyval.node) = block;
   }
#line 5357 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 282:
#line 2764 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      ast_interface_block *block = (ast_interface_block *)(yyvsp[0].node);

      if (!block->default_layout.flags.q.buffer) {
            _mesa_glsl_error(& (yylsp[-1]), state,
                             "memory qualifiers can only be used in the "
                             "declaration of shader storage blocks");
      }
      if (!(yyvsp[-1].type_qualifier).merge_qualifier(& (yylsp[-1]), state, block->layout, false)) {
         YYERROR;
      }
      block->layout = (yyvsp[-1].type_qualifier);
      (yyval.node) = block;
   }
#line 5376 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 283:
#line 2782 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      ast_interface_block *const block = (yyvsp[-1].interface_block);

      if ((yyvsp[-6].type_qualifier).flags.q.uniform) {
         block->default_layout = *state->default_uniform_qualifier;
      } else if ((yyvsp[-6].type_qualifier).flags.q.buffer) {
         block->default_layout = *state->default_shader_storage_qualifier;
      }
      block->block_name = (yyvsp[-5].identifier);
      block->declarations.push_degenerate_list_at_head(& (yyvsp[-3].declarator_list)->link);

      _mesa_ast_process_interface_block(& (yylsp[-6]), state, block, (yyvsp[-6].type_qualifier));

      (yyval.interface_block) = block;
   }
#line 5396 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 284:
#line 2801 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      memset(& (yyval.type_qualifier), 0, sizeof((yyval.type_qualifier)));
      (yyval.type_qualifier).flags.q.in = 1;
   }
#line 5405 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 285:
#line 2806 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      memset(& (yyval.type_qualifier), 0, sizeof((yyval.type_qualifier)));
      (yyval.type_qualifier).flags.q.out = 1;
   }
#line 5414 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 286:
#line 2811 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      memset(& (yyval.type_qualifier), 0, sizeof((yyval.type_qualifier)));
      (yyval.type_qualifier).flags.q.uniform = 1;
   }
#line 5423 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 287:
#line 2816 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      memset(& (yyval.type_qualifier), 0, sizeof((yyval.type_qualifier)));
      (yyval.type_qualifier).flags.q.buffer = 1;
   }
#line 5432 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 288:
#line 2821 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      if (!(yyvsp[-1].type_qualifier).flags.q.patch) {
         _mesa_glsl_error(&(yylsp[-1]), state, "invalid interface qualifier");
      }
      if ((yyvsp[0].type_qualifier).has_auxiliary_storage()) {
         _mesa_glsl_error(&(yylsp[-1]), state, "duplicate patch qualifier");
      }
      (yyval.type_qualifier) = (yyvsp[0].type_qualifier);
      (yyval.type_qualifier).flags.q.patch = 1;
   }
#line 5447 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 289:
#line 2835 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      (yyval.interface_block) = new(state->linalloc) ast_interface_block(NULL, NULL);
   }
#line 5455 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 290:
#line 2839 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      (yyval.interface_block) = new(state->linalloc) ast_interface_block((yyvsp[0].identifier), NULL);
      (yyval.interface_block)->set_location((yylsp[0]));
   }
#line 5464 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 291:
#line 2844 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      (yyval.interface_block) = new(state->linalloc) ast_interface_block((yyvsp[-1].identifier), (yyvsp[0].array_specifier));
      (yyval.interface_block)->set_location_range((yylsp[-1]), (yylsp[0]));
   }
#line 5473 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 292:
#line 2852 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      (yyval.declarator_list) = (yyvsp[0].declarator_list);
      (yyvsp[0].declarator_list)->link.self_link();
   }
#line 5482 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 293:
#line 2857 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      (yyval.declarator_list) = (yyvsp[-1].declarator_list);
      (yyvsp[0].declarator_list)->link.insert_before(& (yyval.declarator_list)->link);
   }
#line 5491 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 294:
#line 2865 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      void *ctx = state->linalloc;
      ast_fully_specified_type *type = (yyvsp[-2].fully_specified_type);
      type->set_location((yylsp[-2]));

      if (type->qualifier.flags.q.attribute) {
         _mesa_glsl_error(& (yylsp[-2]), state,
                          "keyword 'attribute' cannot be used with "
                          "interface block member");
      } else if (type->qualifier.flags.q.varying) {
         _mesa_glsl_error(& (yylsp[-2]), state,
                          "keyword 'varying' cannot be used with "
                          "interface block member");
      }

      (yyval.declarator_list) = new(ctx) ast_declarator_list(type);
      (yyval.declarator_list)->set_location((yylsp[-1]));

      (yyval.declarator_list)->declarations.push_degenerate_list_at_head(& (yyvsp[-1].declaration)->link);
   }
#line 5516 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 295:
#line 2889 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      (yyval.type_qualifier) = (yyvsp[-1].type_qualifier);
      if (!(yyval.type_qualifier).merge_qualifier(& (yylsp[-1]), state, (yyvsp[0].type_qualifier), false, true)) {
         YYERROR;
      }
   }
#line 5527 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 297:
#line 2900 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      (yyval.type_qualifier) = (yyvsp[-1].type_qualifier);
      if (!(yyval.type_qualifier).merge_qualifier(& (yylsp[-1]), state, (yyvsp[0].type_qualifier), false, true)) {
         YYERROR;
      }
   }
#line 5538 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 299:
#line 2911 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      (yyval.type_qualifier) = (yyvsp[-1].type_qualifier);
      if (!(yyval.type_qualifier).merge_qualifier(& (yylsp[-1]), state, (yyvsp[0].type_qualifier), false, true)) {
         YYERROR;
      }
      if (!(yyval.type_qualifier).validate_in_qualifier(& (yylsp[-1]), state)) {
         YYERROR;
      }
   }
#line 5552 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 300:
#line 2921 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      if (!(yyvsp[-2].type_qualifier).validate_in_qualifier(& (yylsp[-2]), state)) {
         YYERROR;
      }
   }
#line 5562 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 301:
#line 2930 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      (yyval.type_qualifier) = (yyvsp[-1].type_qualifier);
      if (!(yyval.type_qualifier).merge_qualifier(& (yylsp[-1]), state, (yyvsp[0].type_qualifier), false, true)) {
         YYERROR;
      }
      if (!(yyval.type_qualifier).validate_out_qualifier(& (yylsp[-1]), state)) {
         YYERROR;
      }
   }
#line 5576 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 302:
#line 2940 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      if (!(yyvsp[-2].type_qualifier).validate_out_qualifier(& (yylsp[-2]), state)) {
         YYERROR;
      }
   }
#line 5586 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 303:
#line 2949 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      (yyval.node) = NULL;
      if (!state->default_uniform_qualifier->
             merge_qualifier(& (yylsp[0]), state, (yyvsp[0].type_qualifier), false)) {
         YYERROR;
      }
      if (!state->default_uniform_qualifier->
             push_to_global(& (yylsp[0]), state)) {
         YYERROR;
      }
   }
#line 5602 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 304:
#line 2961 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      (yyval.node) = NULL;
      if (!state->default_shader_storage_qualifier->
             merge_qualifier(& (yylsp[0]), state, (yyvsp[0].type_qualifier), false)) {
         YYERROR;
      }
      if (!state->default_shader_storage_qualifier->
             push_to_global(& (yylsp[0]), state)) {
         YYERROR;
      }

      /* From the GLSL 4.50 spec, section 4.4.5:
       *
       *     "It is a compile-time error to specify the binding identifier for
       *     the global scope or for block member declarations."
       */
      if (state->default_shader_storage_qualifier->flags.q.explicit_binding) {
         _mesa_glsl_error(& (yylsp[0]), state,
                          "binding qualifier cannot be set for default layout");
      }
   }
#line 5628 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 305:
#line 2983 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      (yyval.node) = NULL;
      if (!(yyvsp[0].type_qualifier).merge_into_in_qualifier(& (yylsp[0]), state, (yyval.node))) {
         YYERROR;
      }
      if (!state->in_qualifier->push_to_global(& (yylsp[0]), state)) {
         YYERROR;
      }
   }
#line 5642 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;

  case 306:
#line 2993 "./glsl/glsl_parser.yy" /* yacc.c:1651  */
    {
      (yyval.node) = NULL;
      if (!(yyvsp[0].type_qualifier).merge_into_out_qualifier(& (yylsp[0]), state, (yyval.node))) {
         YYERROR;
      }
      if (!state->out_qualifier->push_to_global(& (yylsp[0]), state)) {
         YYERROR;
      }
   }
#line 5656 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
    break;


#line 5660 "glsl/glsl_parser.cpp" /* yacc.c:1651  */
      default: break;
    }
  /* User semantic actions sometimes alter yychar, and that requires
     that yytoken be updated with the new translation.  We take the
     approach of translating immediately before every use of yytoken.
     One alternative is translating here after every semantic action,
     but that translation would be missed if the semantic action invokes
     YYABORT, YYACCEPT, or YYERROR immediately after altering yychar or
     if it invokes YYBACKUP.  In the case of YYABORT or YYACCEPT, an
     incorrect destructor might then be invoked immediately.  In the
     case of YYERROR or YYBACKUP, subsequent parser actions might lead
     to an incorrect destructor call or verbose syntax error message
     before the lookahead is translated.  */
  YY_SYMBOL_PRINT ("-> $$ =", yyr1[yyn], &yyval, &yyloc);

  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);

  *++yyvsp = yyval;
  *++yylsp = yyloc;

  /* Now 'shift' the result of the reduction.  Determine what state
     that goes to, based on the state we popped back to and the rule
     number reduced by.  */

  yyn = yyr1[yyn];

  yystate = yypgoto[yyn - YYNTOKENS] + *yyssp;
  if (0 <= yystate && yystate <= YYLAST && yycheck[yystate] == *yyssp)
    yystate = yytable[yystate];
  else
    yystate = yydefgoto[yyn - YYNTOKENS];

  goto yynewstate;


/*--------------------------------------.
| yyerrlab -- here on detecting error.  |
`--------------------------------------*/
yyerrlab:
  /* Make sure we have latest lookahead translation.  See comments at
     user semantic actions for why this is necessary.  */
  yytoken = yychar == YYEMPTY ? YYEMPTY : YYTRANSLATE (yychar);

  /* If not already recovering from an error, report this error.  */
  if (!yyerrstatus)
    {
      ++yynerrs;
#if ! YYERROR_VERBOSE
      yyerror (&yylloc, state, YY_("syntax error"));
#else
# define YYSYNTAX_ERROR yysyntax_error (&yymsg_alloc, &yymsg, \
                                        yyssp, yytoken)
      {
        char const *yymsgp = YY_("syntax error");
        int yysyntax_error_status;
        yysyntax_error_status = YYSYNTAX_ERROR;
        if (yysyntax_error_status == 0)
          yymsgp = yymsg;
        else if (yysyntax_error_status == 1)
          {
            if (yymsg != yymsgbuf)
              YYSTACK_FREE (yymsg);
            yymsg = (char *) YYSTACK_ALLOC (yymsg_alloc);
            if (!yymsg)
              {
                yymsg = yymsgbuf;
                yymsg_alloc = sizeof yymsgbuf;
                yysyntax_error_status = 2;
              }
            else
              {
                yysyntax_error_status = YYSYNTAX_ERROR;
                yymsgp = yymsg;
              }
          }
        yyerror (&yylloc, state, yymsgp);
        if (yysyntax_error_status == 2)
          goto yyexhaustedlab;
      }
# undef YYSYNTAX_ERROR
#endif
    }

  yyerror_range[1] = yylloc;

  if (yyerrstatus == 3)
    {
      /* If just tried and failed to reuse lookahead token after an
         error, discard it.  */

      if (yychar <= YYEOF)
        {
          /* Return failure if at end of input.  */
          if (yychar == YYEOF)
            YYABORT;
        }
      else
        {
          yydestruct ("Error: discarding",
                      yytoken, &yylval, &yylloc, state);
          yychar = YYEMPTY;
        }
    }

  /* Else will try to reuse lookahead token after shifting the error
     token.  */
  goto yyerrlab1;


/*---------------------------------------------------.
| yyerrorlab -- error raised explicitly by YYERROR.  |
`---------------------------------------------------*/
yyerrorlab:

  /* Pacify compilers like GCC when the user code never invokes
     YYERROR and the label yyerrorlab therefore never appears in user
     code.  */
  if (/*CONSTCOND*/ 0)
     goto yyerrorlab;

  /* Do not reclaim the symbols of the rule whose action triggered
     this YYERROR.  */
  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);
  yystate = *yyssp;
  goto yyerrlab1;


/*-------------------------------------------------------------.
| yyerrlab1 -- common code for both syntax error and YYERROR.  |
`-------------------------------------------------------------*/
yyerrlab1:
  yyerrstatus = 3;      /* Each real token shifted decrements this.  */

  for (;;)
    {
      yyn = yypact[yystate];
      if (!yypact_value_is_default (yyn))
        {
          yyn += YYTERROR;
          if (0 <= yyn && yyn <= YYLAST && yycheck[yyn] == YYTERROR)
            {
              yyn = yytable[yyn];
              if (0 < yyn)
                break;
            }
        }

      /* Pop the current state because it cannot handle the error token.  */
      if (yyssp == yyss)
        YYABORT;

      yyerror_range[1] = *yylsp;
      yydestruct ("Error: popping",
                  yystos[yystate], yyvsp, yylsp, state);
      YYPOPSTACK (1);
      yystate = *yyssp;
      YY_STACK_PRINT (yyss, yyssp);
    }

  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END

  yyerror_range[2] = yylloc;
  /* Using YYLLOC is tempting, but would change the location of
     the lookahead.  YYLOC is available though.  */
  YYLLOC_DEFAULT (yyloc, yyerror_range, 2);
  *++yylsp = yyloc;

  /* Shift the error token.  */
  YY_SYMBOL_PRINT ("Shifting", yystos[yyn], yyvsp, yylsp);

  yystate = yyn;
  goto yynewstate;


/*-------------------------------------.
| yyacceptlab -- YYACCEPT comes here.  |
`-------------------------------------*/
yyacceptlab:
  yyresult = 0;
  goto yyreturn;

/*-----------------------------------.
| yyabortlab -- YYABORT comes here.  |
`-----------------------------------*/
yyabortlab:
  yyresult = 1;
  goto yyreturn;

#if !defined yyoverflow || YYERROR_VERBOSE
/*-------------------------------------------------.
| yyexhaustedlab -- memory exhaustion comes here.  |
`-------------------------------------------------*/
yyexhaustedlab:
  yyerror (&yylloc, state, YY_("memory exhausted"));
  yyresult = 2;
  /* Fall through.  */
#endif

yyreturn:
  if (yychar != YYEMPTY)
    {
      /* Make sure we have latest lookahead translation.  See comments at
         user semantic actions for why this is necessary.  */
      yytoken = YYTRANSLATE (yychar);
      yydestruct ("Cleanup: discarding lookahead",
                  yytoken, &yylval, &yylloc, state);
    }
  /* Do not reclaim the symbols of the rule whose action triggered
     this YYABORT or YYACCEPT.  */
  YYPOPSTACK (yylen);
  YY_STACK_PRINT (yyss, yyssp);
  while (yyssp != yyss)
    {
      yydestruct ("Cleanup: popping",
                  yystos[*yyssp], yyvsp, yylsp, state);
      YYPOPSTACK (1);
    }
#ifndef yyoverflow
  if (yyss != yyssa)
    YYSTACK_FREE (yyss);
#endif
#if YYERROR_VERBOSE
  if (yymsg != yymsgbuf)
    YYSTACK_FREE (yymsg);
#endif
  return yyresult;
}
