/* This is file is generated automatically. Don't edit!  */
/*
 * XML DRI client-side driver configuration
 * Copyright (C) 2003 Felix Kuehling
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * FELIX KUEHLING, OR ANY OTHER CONTRIBUTORS BE LIABLE FOR ANY CLAIM, 
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR 
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE 
 * OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 * 
 */
/**
 * \file t_options.h
 * \brief Templates of common options
 * \author Felix Kuehling
 *
 * This file defines macros for common options that can be used to
 * construct driConfigOptions in the drivers. This file is only a
 * template containing English descriptions for options wrapped in
 * gettext(). xgettext can be used to extract translatable
 * strings. These strings can then be translated by anyone familiar
 * with GNU gettext. gen_xmlpool.py takes this template and fills in
 * all the translations. The result (options.h) is included by
 * xmlpool.h which in turn can be included by drivers.
 *
 * The macros used to describe otions in this file are defined in
 * ../xmlpool.h.
 */

/* This is needed for xgettext to extract translatable strings.
 * gen_xmlpool.py will discard this line. */
/* #include <libintl.h>
 * commented out by gen_xmlpool.py */

/*
 * predefined option sections and options with multi-lingual descriptions
 */


/**
 * \brief Debugging options
 */
#define DRI_CONF_SECTION_DEBUG \
DRI_CONF_SECTION_BEGIN \
        DRI_CONF_DESC(en,"Debugging") \
        DRI_CONF_DESC(ca,"Depuració") \
        DRI_CONF_DESC(de,"Fehlersuche") \
        DRI_CONF_DESC(es,"Depuración") \
        DRI_CONF_DESC(nl,"Debuggen") \
        DRI_CONF_DESC(fr,"Debogage") \
        DRI_CONF_DESC(sv,"Felsökning")

#define DRI_CONF_ALWAYS_FLUSH_BATCH(def) \
DRI_CONF_OPT_BEGIN_B(always_flush_batch, def) \
        DRI_CONF_DESC(en,"Enable flushing batchbuffer after each draw call") \
        DRI_CONF_DESC(ca,"Habilita el buidatge del batchbuffer després de cada trucada de dibuix") \
        DRI_CONF_DESC(de,"Aktiviere sofortige Leerung des Stapelpuffers nach jedem Zeichenaufruf") \
        DRI_CONF_DESC(es,"Habilitar vaciado del batchbuffer después de cada llamada de dibujo") \
        DRI_CONF_DESC(nl,"Enable flushing batchbuffer after each draw call") \
        DRI_CONF_DESC(fr,"Enable flushing batchbuffer after each draw call") \
        DRI_CONF_DESC(sv,"Enable flushing batchbuffer after each draw call") \
DRI_CONF_OPT_END

#define DRI_CONF_ALWAYS_FLUSH_CACHE(def) \
DRI_CONF_OPT_BEGIN_B(always_flush_cache, def) \
        DRI_CONF_DESC(en,"Enable flushing GPU caches with each draw call") \
        DRI_CONF_DESC(ca,"Habilita el buidatge de les memòries cau de GPU amb cada trucada de dibuix") \
        DRI_CONF_DESC(de,"Aktiviere sofortige Leerung der GPU-Zwischenspeicher mit jedem Zeichenaufruf") \
        DRI_CONF_DESC(es,"Habilitar vaciado de los cachés GPU con cada llamada de dibujo") \
        DRI_CONF_DESC(nl,"Enable flushing GPU caches with each draw call") \
        DRI_CONF_DESC(fr,"Enable flushing GPU caches with each draw call") \
        DRI_CONF_DESC(sv,"Enable flushing GPU caches with each draw call") \
DRI_CONF_OPT_END

#define DRI_CONF_DISABLE_THROTTLING(def) \
DRI_CONF_OPT_BEGIN_B(disable_throttling, def) \
        DRI_CONF_DESC(en,"Disable throttling on first batch after flush") \
        DRI_CONF_DESC(ca,"Deshabilita la regulació en el primer lot després de buidar") \
        DRI_CONF_DESC(de,"Disable throttling on first batch after flush") \
        DRI_CONF_DESC(es,"Deshabilitar regulación del primer lote después de vaciar") \
        DRI_CONF_DESC(nl,"Disable throttling on first batch after flush") \
        DRI_CONF_DESC(fr,"Disable throttling on first batch after flush") \
        DRI_CONF_DESC(sv,"Disable throttling on first batch after flush") \
DRI_CONF_OPT_END

#define DRI_CONF_FORCE_GLSL_EXTENSIONS_WARN(def) \
DRI_CONF_OPT_BEGIN_B(force_glsl_extensions_warn, def) \
        DRI_CONF_DESC(en,"Force GLSL extension default behavior to 'warn'") \
        DRI_CONF_DESC(ca,"Força que el comportament per defecte de les extensions GLSL sigui 'warn'") \
        DRI_CONF_DESC(de,"Force GLSL extension default behavior to 'warn'") \
        DRI_CONF_DESC(es,"Forzar que el comportamiento por defecto de las extensiones GLSL sea 'warn'") \
        DRI_CONF_DESC(nl,"Force GLSL extension default behavior to 'warn'") \
        DRI_CONF_DESC(fr,"Force GLSL extension default behavior to 'warn'") \
        DRI_CONF_DESC(sv,"Force GLSL extension default behavior to 'warn'") \
DRI_CONF_OPT_END

#define DRI_CONF_DISABLE_BLEND_FUNC_EXTENDED(def) \
DRI_CONF_OPT_BEGIN_B(disable_blend_func_extended, def) \
        DRI_CONF_DESC(en,"Disable dual source blending") \
        DRI_CONF_DESC(ca,"Deshabilita la barreja de font dual") \
        DRI_CONF_DESC(de,"Disable dual source blending") \
        DRI_CONF_DESC(es,"Deshabilitar mezcla de fuente dual") \
        DRI_CONF_DESC(nl,"Disable dual source blending") \
        DRI_CONF_DESC(fr,"Disable dual source blending") \
        DRI_CONF_DESC(sv,"Disable dual source blending") \
DRI_CONF_OPT_END

#define DRI_CONF_DUAL_COLOR_BLEND_BY_LOCATION(def) \
DRI_CONF_OPT_BEGIN_B(dual_color_blend_by_location, def) \
        DRI_CONF_DESC(en,"Identify dual color blending sources by location rather than index") \
        DRI_CONF_DESC(ca,"Identify dual color blending sources by location rather than index") \
        DRI_CONF_DESC(de,"Identify dual color blending sources by location rather than index") \
        DRI_CONF_DESC(es,"Identify dual color blending sources by location rather than index") \
        DRI_CONF_DESC(nl,"Identify dual color blending sources by location rather than index") \
        DRI_CONF_DESC(fr,"Identify dual color blending sources by location rather than index") \
        DRI_CONF_DESC(sv,"Identify dual color blending sources by location rather than index") \
DRI_CONF_OPT_END

#define DRI_CONF_DISABLE_GLSL_LINE_CONTINUATIONS(def) \
DRI_CONF_OPT_BEGIN_B(disable_glsl_line_continuations, def) \
        DRI_CONF_DESC(en,"Disable backslash-based line continuations in GLSL source") \
        DRI_CONF_DESC(ca,"Deshabilita les continuacions de línia basades en barra invertida en la font GLSL") \
        DRI_CONF_DESC(de,"Disable backslash-based line continuations in GLSL source") \
        DRI_CONF_DESC(es,"Deshabilitar continuaciones de línea basadas en barra inversa en el código GLSL") \
        DRI_CONF_DESC(nl,"Disable backslash-based line continuations in GLSL source") \
        DRI_CONF_DESC(fr,"Disable backslash-based line continuations in GLSL source") \
        DRI_CONF_DESC(sv,"Disable backslash-based line continuations in GLSL source") \
DRI_CONF_OPT_END

#define DRI_CONF_FORCE_GLSL_VERSION(def) \
DRI_CONF_OPT_BEGIN_V(force_glsl_version, int, def, "0:999") \
        DRI_CONF_DESC(en,"Force a default GLSL version for shaders that lack an explicit #version line") \
        DRI_CONF_DESC(ca,"Força una versió GLSL per defecte en els shaders als quals lis manca una línia #version explícita") \
        DRI_CONF_DESC(de,"Force a default GLSL version for shaders that lack an explicit #version line") \
        DRI_CONF_DESC(es,"Forzar una versión de GLSL por defecto en los shaders a los cuales les falta una línea #version explícita") \
        DRI_CONF_DESC(nl,"Force a default GLSL version for shaders that lack an explicit #version line") \
        DRI_CONF_DESC(fr,"Force a default GLSL version for shaders that lack an explicit #version line") \
        DRI_CONF_DESC(sv,"Force a default GLSL version for shaders that lack an explicit #version line") \
DRI_CONF_OPT_END

#define DRI_CONF_ALLOW_GLSL_EXTENSION_DIRECTIVE_MIDSHADER(def) \
DRI_CONF_OPT_BEGIN_B(allow_glsl_extension_directive_midshader, def) \
        DRI_CONF_DESC(en,"Allow GLSL #extension directives in the middle of shaders") \
        DRI_CONF_DESC(ca,"Permet les directives #extension GLSL en el mitjà dels shaders") \
        DRI_CONF_DESC(de,"Allow GLSL #extension directives in the middle of shaders") \
        DRI_CONF_DESC(es,"Permite directivas #extension GLSL en medio de los shaders") \
        DRI_CONF_DESC(nl,"Allow GLSL #extension directives in the middle of shaders") \
        DRI_CONF_DESC(fr,"Allow GLSL #extension directives in the middle of shaders") \
        DRI_CONF_DESC(sv,"Allow GLSL #extension directives in the middle of shaders") \
DRI_CONF_OPT_END

#define DRI_CONF_ALLOW_GLSL_BUILTIN_CONST_EXPRESSION(def) \
DRI_CONF_OPT_BEGIN_B(allow_glsl_builtin_const_expression, def) \
        DRI_CONF_DESC(en,"Allow builtins as part of constant expressions") \
        DRI_CONF_DESC(ca,"Allow builtins as part of constant expressions") \
        DRI_CONF_DESC(de,"Allow builtins as part of constant expressions") \
        DRI_CONF_DESC(es,"Allow builtins as part of constant expressions") \
        DRI_CONF_DESC(nl,"Allow builtins as part of constant expressions") \
        DRI_CONF_DESC(fr,"Allow builtins as part of constant expressions") \
        DRI_CONF_DESC(sv,"Allow builtins as part of constant expressions") \
DRI_CONF_OPT_END

#define DRI_CONF_ALLOW_GLSL_RELAXED_ES(def) \
DRI_CONF_OPT_BEGIN_B(allow_glsl_relaxed_es, def) \
        DRI_CONF_DESC(en,"Allow some relaxation of GLSL ES shader restrictions") \
        DRI_CONF_DESC(ca,"Allow some relaxation of GLSL ES shader restrictions") \
        DRI_CONF_DESC(de,"Allow some relaxation of GLSL ES shader restrictions") \
        DRI_CONF_DESC(es,"Allow some relaxation of GLSL ES shader restrictions") \
        DRI_CONF_DESC(nl,"Allow some relaxation of GLSL ES shader restrictions") \
        DRI_CONF_DESC(fr,"Allow some relaxation of GLSL ES shader restrictions") \
        DRI_CONF_DESC(sv,"Allow some relaxation of GLSL ES shader restrictions") \
DRI_CONF_OPT_END

#define DRI_CONF_ALLOW_GLSL_BUILTIN_VARIABLE_REDECLARATION(def) \
DRI_CONF_OPT_BEGIN_B(allow_glsl_builtin_variable_redeclaration, def) \
        DRI_CONF_DESC(en,"Allow GLSL built-in variables to be redeclared verbatim") \
        DRI_CONF_DESC(ca,"Allow GLSL built-in variables to be redeclared verbatim") \
        DRI_CONF_DESC(de,"Allow GLSL built-in variables to be redeclared verbatim") \
        DRI_CONF_DESC(es,"Allow GLSL built-in variables to be redeclared verbatim") \
        DRI_CONF_DESC(nl,"Allow GLSL built-in variables to be redeclared verbatim") \
        DRI_CONF_DESC(fr,"Allow GLSL built-in variables to be redeclared verbatim") \
        DRI_CONF_DESC(sv,"Allow GLSL built-in variables to be redeclared verbatim") \
DRI_CONF_OPT_END

#define DRI_CONF_ALLOW_HIGHER_COMPAT_VERSION(def) \
DRI_CONF_OPT_BEGIN_B(allow_higher_compat_version, def) \
        DRI_CONF_DESC(en,"Allow a higher compat profile (version 3.1+) for apps that request it") \
        DRI_CONF_DESC(ca,"Allow a higher compat profile (version 3.1+) for apps that request it") \
        DRI_CONF_DESC(de,"Allow a higher compat profile (version 3.1+) for apps that request it") \
        DRI_CONF_DESC(es,"Allow a higher compat profile (version 3.1+) for apps that request it") \
        DRI_CONF_DESC(nl,"Allow a higher compat profile (version 3.1+) for apps that request it") \
        DRI_CONF_DESC(fr,"Allow a higher compat profile (version 3.1+) for apps that request it") \
        DRI_CONF_DESC(sv,"Allow a higher compat profile (version 3.1+) for apps that request it") \
DRI_CONF_OPT_END

#define DRI_CONF_FORCE_GLSL_ABS_SQRT(def) \
DRI_CONF_OPT_BEGIN_B(force_glsl_abs_sqrt, def) \
        DRI_CONF_DESC(en,"Force computing the absolute value for sqrt() and inversesqrt()") \
        DRI_CONF_DESC(ca,"Force computing the absolute value for sqrt() and inversesqrt()") \
        DRI_CONF_DESC(de,"Force computing the absolute value for sqrt() and inversesqrt()") \
        DRI_CONF_DESC(es,"Force computing the absolute value for sqrt() and inversesqrt()") \
        DRI_CONF_DESC(nl,"Force computing the absolute value for sqrt() and inversesqrt()") \
        DRI_CONF_DESC(fr,"Force computing the absolute value for sqrt() and inversesqrt()") \
        DRI_CONF_DESC(sv,"Force computing the absolute value for sqrt() and inversesqrt()") \
DRI_CONF_OPT_END

#define DRI_CONF_GLSL_CORRECT_DERIVATIVES_AFTER_DISCARD(def) \
DRI_CONF_OPT_BEGIN_B(glsl_correct_derivatives_after_discard, def) \
        DRI_CONF_DESC(en,"Implicit and explicit derivatives after a discard behave as if the discard didn't happen") \
        DRI_CONF_DESC(ca,"Implicit and explicit derivatives after a discard behave as if the discard didn't happen") \
        DRI_CONF_DESC(de,"Implicit and explicit derivatives after a discard behave as if the discard didn't happen") \
        DRI_CONF_DESC(es,"Implicit and explicit derivatives after a discard behave as if the discard didn't happen") \
        DRI_CONF_DESC(nl,"Implicit and explicit derivatives after a discard behave as if the discard didn't happen") \
        DRI_CONF_DESC(fr,"Implicit and explicit derivatives after a discard behave as if the discard didn't happen") \
        DRI_CONF_DESC(sv,"Implicit and explicit derivatives after a discard behave as if the discard didn't happen") \
DRI_CONF_OPT_END

#define DRI_CONF_ALLOW_GLSL_CROSS_STAGE_INTERPOLATION_MISMATCH(def) \
DRI_CONF_OPT_BEGIN_B(allow_glsl_cross_stage_interpolation_mismatch, def) \
        DRI_CONF_DESC(en,"Allow interpolation qualifier mismatch across shader stages") \
        DRI_CONF_DESC(ca,"Allow interpolation qualifier mismatch across shader stages") \
        DRI_CONF_DESC(de,"Allow interpolation qualifier mismatch across shader stages") \
        DRI_CONF_DESC(es,"Allow interpolation qualifier mismatch across shader stages") \
        DRI_CONF_DESC(nl,"Allow interpolation qualifier mismatch across shader stages") \
        DRI_CONF_DESC(fr,"Allow interpolation qualifier mismatch across shader stages") \
        DRI_CONF_DESC(sv,"Allow interpolation qualifier mismatch across shader stages") \
DRI_CONF_OPT_END

#define DRI_CONF_ALLOW_GLSL_LAYOUT_QUALIFIER_ON_FUNCTION_PARAMETERS(def) \
DRI_CONF_OPT_BEGIN_B(allow_glsl_layout_qualifier_on_function_parameters, def) \
        DRI_CONF_DESC(en,"Allow layout qualifiers on function parameters.") \
        DRI_CONF_DESC(ca,"Allow layout qualifiers on function parameters.") \
        DRI_CONF_DESC(de,"Allow layout qualifiers on function parameters.") \
        DRI_CONF_DESC(es,"Allow layout qualifiers on function parameters.") \
        DRI_CONF_DESC(nl,"Allow layout qualifiers on function parameters.") \
        DRI_CONF_DESC(fr,"Allow layout qualifiers on function parameters.") \
        DRI_CONF_DESC(sv,"Allow layout qualifiers on function parameters.") \
DRI_CONF_OPT_END

#define DRI_CONF_FORCE_COMPAT_PROFILE(def) \
DRI_CONF_OPT_BEGIN_B(force_compat_profile, def) \
        DRI_CONF_DESC(en,"Force an OpenGL compatibility context") \
        DRI_CONF_DESC(ca,"Force an OpenGL compatibility context") \
        DRI_CONF_DESC(de,"Force an OpenGL compatibility context") \
        DRI_CONF_DESC(es,"Force an OpenGL compatibility context") \
        DRI_CONF_DESC(nl,"Force an OpenGL compatibility context") \
        DRI_CONF_DESC(fr,"Force an OpenGL compatibility context") \
        DRI_CONF_DESC(sv,"Force an OpenGL compatibility context") \
DRI_CONF_OPT_END

/**
 * \brief Image quality-related options
 */
#define DRI_CONF_SECTION_QUALITY \
DRI_CONF_SECTION_BEGIN \
        DRI_CONF_DESC(en,"Image Quality") \
        DRI_CONF_DESC(ca,"Qualitat d'imatge") \
        DRI_CONF_DESC(de,"Bildqualität") \
        DRI_CONF_DESC(es,"Calidad de imagen") \
        DRI_CONF_DESC(nl,"Beeldkwaliteit") \
        DRI_CONF_DESC(fr,"Qualité d'image") \
        DRI_CONF_DESC(sv,"Bildkvalitet")

#define DRI_CONF_PRECISE_TRIG(def) \
DRI_CONF_OPT_BEGIN_B(precise_trig, def) \
        DRI_CONF_DESC(en,"Prefer accuracy over performance in trig functions") \
        DRI_CONF_DESC(ca,"Prefer accuracy over performance in trig functions") \
        DRI_CONF_DESC(de,"Prefer accuracy over performance in trig functions") \
        DRI_CONF_DESC(es,"Prefer accuracy over performance in trig functions") \
        DRI_CONF_DESC(nl,"Prefer accuracy over performance in trig functions") \
        DRI_CONF_DESC(fr,"Prefer accuracy over performance in trig functions") \
        DRI_CONF_DESC(sv,"Prefer accuracy over performance in trig functions") \
DRI_CONF_OPT_END

#define DRI_CONF_PP_CELSHADE(def) \
DRI_CONF_OPT_BEGIN_V(pp_celshade,enum,def,"0:1") \
        DRI_CONF_DESC(en,"A post-processing filter to cel-shade the output") \
        DRI_CONF_DESC(ca,"Un filtre de postprocessament per a aplicar cel shading a la sortida") \
        DRI_CONF_DESC(de,"Nachbearbeitungsfilter für Cell Shading") \
        DRI_CONF_DESC(es,"Un filtro de postprocesamiento para aplicar cel shading a la salida") \
        DRI_CONF_DESC(nl,"A post-processing filter to cel-shade the output") \
        DRI_CONF_DESC(fr,"A post-processing filter to cel-shade the output") \
        DRI_CONF_DESC(sv,"A post-processing filter to cel-shade the output") \
DRI_CONF_OPT_END

#define DRI_CONF_PP_NORED(def) \
DRI_CONF_OPT_BEGIN_V(pp_nored,enum,def,"0:1") \
        DRI_CONF_DESC(en,"A post-processing filter to remove the red channel") \
        DRI_CONF_DESC(ca,"Un filtre de postprocessament per a eliminar el canal vermell") \
        DRI_CONF_DESC(de,"Nachbearbeitungsfilter zum Entfernen des Rotkanals") \
        DRI_CONF_DESC(es,"Un filtro de postprocesamiento para eliminar el canal rojo") \
        DRI_CONF_DESC(nl,"A post-processing filter to remove the red channel") \
        DRI_CONF_DESC(fr,"A post-processing filter to remove the red channel") \
        DRI_CONF_DESC(sv,"A post-processing filter to remove the red channel") \
DRI_CONF_OPT_END

#define DRI_CONF_PP_NOGREEN(def) \
DRI_CONF_OPT_BEGIN_V(pp_nogreen,enum,def,"0:1") \
        DRI_CONF_DESC(en,"A post-processing filter to remove the green channel") \
        DRI_CONF_DESC(ca,"Un filtre de postprocessament per a eliminar el canal verd") \
        DRI_CONF_DESC(de,"Nachbearbeitungsfilter zum Entfernen des Grünkanals") \
        DRI_CONF_DESC(es,"Un filtro de postprocesamiento para eliminar el canal verde") \
        DRI_CONF_DESC(nl,"A post-processing filter to remove the green channel") \
        DRI_CONF_DESC(fr,"A post-processing filter to remove the green channel") \
        DRI_CONF_DESC(sv,"A post-processing filter to remove the green channel") \
DRI_CONF_OPT_END

#define DRI_CONF_PP_NOBLUE(def) \
DRI_CONF_OPT_BEGIN_V(pp_noblue,enum,def,"0:1") \
        DRI_CONF_DESC(en,"A post-processing filter to remove the blue channel") \
        DRI_CONF_DESC(ca,"Un filtre de postprocessament per a eliminar el canal blau") \
        DRI_CONF_DESC(de,"Nachbearbeitungsfilter zum Entfernen des Blaukanals") \
        DRI_CONF_DESC(es,"Un filtro de postprocesamiento para eliminar el canal azul") \
        DRI_CONF_DESC(nl,"A post-processing filter to remove the blue channel") \
        DRI_CONF_DESC(fr,"A post-processing filter to remove the blue channel") \
        DRI_CONF_DESC(sv,"A post-processing filter to remove the blue channel") \
DRI_CONF_OPT_END

#define DRI_CONF_PP_JIMENEZMLAA(def,min,max) \
DRI_CONF_OPT_BEGIN_V(pp_jimenezmlaa,int,def, # min ":" # max ) \
        DRI_CONF_DESC(en,"Morphological anti-aliasing based on Jimenez\' MLAA. 0 to disable, 8 for default quality") \
        DRI_CONF_DESC(ca,"Antialiàsing morfològic basat en el MLAA de Jimenez. 0 per deshabilitar, 8 per qualitat per defecte") \
        DRI_CONF_DESC(de,"Morphologische Kantenglättung (Anti-Aliasing) basierend auf Jimenez' MLAA. 0 für deaktiviert, 8 für Standardqualität") \
        DRI_CONF_DESC(es,"Antialiasing morfológico basado en el MLAA de Jimenez. 0 para deshabilitar, 8 para calidad por defecto") \
        DRI_CONF_DESC(nl,"Morphological anti-aliasing based on Jimenez\' MLAA. 0 to disable, 8 for default quality") \
        DRI_CONF_DESC(fr,"Morphological anti-aliasing based on Jimenez\' MLAA. 0 to disable, 8 for default quality") \
        DRI_CONF_DESC(sv,"Morphological anti-aliasing based on Jimenez\' MLAA. 0 to disable, 8 for default quality") \
DRI_CONF_OPT_END

#define DRI_CONF_PP_JIMENEZMLAA_COLOR(def,min,max) \
DRI_CONF_OPT_BEGIN_V(pp_jimenezmlaa_color,int,def, # min ":" # max ) \
        DRI_CONF_DESC(en,"Morphological anti-aliasing based on Jimenez\' MLAA. 0 to disable, 8 for default quality. Color version, usable with 2d GL apps") \
        DRI_CONF_DESC(ca,"Antialiàsing morfològic basat en el MLAA de Jimenez. 0 per deshabilitar, 8 per qualitat per defecte. Versió en color, utilitzable amb les aplicacions GL 2D") \
        DRI_CONF_DESC(de,"Morphologische Kantenglättung (Anti-Aliasing) basierend auf Jimenez' MLAA. 0 für deaktiviert, 8 für Standardqualität. Farbversion, für 2D-Anwendungen") \
        DRI_CONF_DESC(es,"Antialiasing morfológico basado en el MLAA de Jimenez. 0 para deshabilitar, 8 para calidad por defecto. Versión en color, usable con aplicaciones GL 2D") \
        DRI_CONF_DESC(nl,"Morphological anti-aliasing based on Jimenez\' MLAA. 0 to disable, 8 for default quality. Color version, usable with 2d GL apps") \
        DRI_CONF_DESC(fr,"Morphological anti-aliasing based on Jimenez\' MLAA. 0 to disable, 8 for default quality. Color version, usable with 2d GL apps") \
        DRI_CONF_DESC(sv,"Morphological anti-aliasing based on Jimenez\' MLAA. 0 to disable, 8 for default quality. Color version, usable with 2d GL apps") \
DRI_CONF_OPT_END



/**
 * \brief Performance-related options
 */
#define DRI_CONF_SECTION_PERFORMANCE \
DRI_CONF_SECTION_BEGIN \
        DRI_CONF_DESC(en,"Performance") \
        DRI_CONF_DESC(ca,"Rendiment") \
        DRI_CONF_DESC(de,"Leistung") \
        DRI_CONF_DESC(es,"Rendimiento") \
        DRI_CONF_DESC(nl,"Prestatie") \
        DRI_CONF_DESC(fr,"Performance") \
        DRI_CONF_DESC(sv,"Prestanda")

#define DRI_CONF_VBLANK_NEVER 0
#define DRI_CONF_VBLANK_DEF_INTERVAL_0 1
#define DRI_CONF_VBLANK_DEF_INTERVAL_1 2
#define DRI_CONF_VBLANK_ALWAYS_SYNC 3
#define DRI_CONF_VBLANK_MODE(def) \
DRI_CONF_OPT_BEGIN_V(vblank_mode,enum,def,"0:3") \
        DRI_CONF_DESC_BEGIN(en,"Synchronization with vertical refresh (swap intervals)") \
                DRI_CONF_ENUM(0,"Never synchronize with vertical refresh, ignore application's choice") \
                DRI_CONF_ENUM(1,"Initial swap interval 0, obey application's choice") \
                DRI_CONF_ENUM(2,"Initial swap interval 1, obey application's choice") \
                DRI_CONF_ENUM(3,"Always synchronize with vertical refresh, application chooses the minimum swap interval") \
        DRI_CONF_DESC_END \
        DRI_CONF_DESC_BEGIN(ca,"Sincronització amb refresc vertical (intervals d'intercanvi)") \
                DRI_CONF_ENUM(0,"Mai sincronitzis amb el refresc vertical, ignora l'elecció de l'aplicació") \
                DRI_CONF_ENUM(1,"Interval d'intercanvi inicial 0, obeeix l'elecció de l'aplicació") \
                DRI_CONF_ENUM(2,"Interval d'intercanvi inicial 1, obeeix l'elecció de l'aplicació") \
                DRI_CONF_ENUM(3,"Sempre sincronitza amb el refresc vertical, l'aplicació tria l'interval mínim d'intercanvi") \
        DRI_CONF_DESC_END \
        DRI_CONF_DESC_BEGIN(de,"Synchronisation mit der vertikalen Bildwiederholung") \
                DRI_CONF_ENUM(0,"Niemals mit der Bildwiederholung synchronisieren, Anweisungen der Anwendung ignorieren") \
                DRI_CONF_ENUM(1,"Initiales Bildinterval 0, Anweisungen der Anwendung gehorchen") \
                DRI_CONF_ENUM(2,"Initiales Bildinterval 1, Anweisungen der Anwendung gehorchen") \
                DRI_CONF_ENUM(3,"Immer mit der Bildwiederholung synchronisieren, Anwendung wählt das minimale Bildintervall") \
        DRI_CONF_DESC_END \
        DRI_CONF_DESC_BEGIN(es,"Sincronización con el refresco vertical (intervalos de intercambio)") \
                DRI_CONF_ENUM(0,"No sincronizar nunca con el refresco vertical, ignorar la elección de la aplicación") \
                DRI_CONF_ENUM(1,"Intervalo de intercambio inicial 0, obedecer la elección de la aplicación") \
                DRI_CONF_ENUM(2,"Intervalo de intercambio inicial 1, obedecer la elección de la aplicación") \
                DRI_CONF_ENUM(3,"Sincronizar siempre con el refresco vertical, la aplicación elige el intervalo de intercambio mínimo") \
        DRI_CONF_DESC_END \
        DRI_CONF_DESC_BEGIN(nl,"Synchronisatie met verticale verversing (interval omwisselen)") \
                DRI_CONF_ENUM(0,"Nooit synchroniseren met verticale verversing, negeer de keuze van de applicatie") \
                DRI_CONF_ENUM(1,"Initïeel omwisselingsinterval 0, honoreer de keuze van de applicatie") \
                DRI_CONF_ENUM(2,"Initïeel omwisselingsinterval 1, honoreer de keuze van de applicatie") \
                DRI_CONF_ENUM(3,"Synchroniseer altijd met verticale verversing, de applicatie kiest het minimum omwisselingsinterval") \
        DRI_CONF_DESC_END \
        DRI_CONF_DESC_BEGIN(fr,"Synchronisation de l'affichage avec le balayage vertical") \
                DRI_CONF_ENUM(0,"Ne jamais synchroniser avec le balayage vertical, ignorer le choix de l'application") \
                DRI_CONF_ENUM(1,"Ne pas synchroniser avec le balayage vertical par défaut, mais obéir au choix de l'application") \
                DRI_CONF_ENUM(2,"Synchroniser avec le balayage vertical par défaut, mais obéir au choix de l'application") \
                DRI_CONF_ENUM(3,"Toujours synchroniser avec le balayage vertical, l'application choisit l'intervalle minimal") \
        DRI_CONF_DESC_END \
        DRI_CONF_DESC_BEGIN(sv,"Synkronisering med vertikal uppdatering (växlingsintervall)") \
                DRI_CONF_ENUM(0,"Synkronisera aldrig med vertikal uppdatering, ignorera programmets val") \
                DRI_CONF_ENUM(1,"Initialt växlingsintervall 0, följ programmets val") \
                DRI_CONF_ENUM(2,"Initialt växlingsintervall 1, följ programmets val") \
                DRI_CONF_ENUM(3,"Synkronisera alltid med vertikal uppdatering, programmet väljer den minsta växlingsintervallen") \
        DRI_CONF_DESC_END \
DRI_CONF_OPT_END

#define DRI_CONF_MESA_GLTHREAD(def) \
DRI_CONF_OPT_BEGIN_B(mesa_glthread, def) \
        DRI_CONF_DESC(en,"Enable offloading GL driver work to a separate thread") \
        DRI_CONF_DESC(ca,"Enable offloading GL driver work to a separate thread") \
        DRI_CONF_DESC(de,"Enable offloading GL driver work to a separate thread") \
        DRI_CONF_DESC(es,"Enable offloading GL driver work to a separate thread") \
        DRI_CONF_DESC(nl,"Enable offloading GL driver work to a separate thread") \
        DRI_CONF_DESC(fr,"Enable offloading GL driver work to a separate thread") \
        DRI_CONF_DESC(sv,"Enable offloading GL driver work to a separate thread") \
DRI_CONF_OPT_END

#define DRI_CONF_MESA_NO_ERROR(def) \
DRI_CONF_OPT_BEGIN_B(mesa_no_error, def) \
        DRI_CONF_DESC(en,"Disable GL driver error checking") \
        DRI_CONF_DESC(ca,"Disable GL driver error checking") \
        DRI_CONF_DESC(de,"Disable GL driver error checking") \
        DRI_CONF_DESC(es,"Disable GL driver error checking") \
        DRI_CONF_DESC(nl,"Disable GL driver error checking") \
        DRI_CONF_DESC(fr,"Disable GL driver error checking") \
        DRI_CONF_DESC(sv,"Disable GL driver error checking") \
DRI_CONF_OPT_END

#define DRI_CONF_DISABLE_EXT_BUFFER_AGE(def) \
DRI_CONF_OPT_BEGIN_B(glx_disable_ext_buffer_age, def) \
   DRI_CONF_DESC(en, "Disable the GLX_EXT_buffer_age extension") \
   DRI_CONF_DESC(ca, "Disable the GLX_EXT_buffer_age extension") \
   DRI_CONF_DESC(de, "Disable the GLX_EXT_buffer_age extension") \
   DRI_CONF_DESC(es, "Disable the GLX_EXT_buffer_age extension") \
   DRI_CONF_DESC(nl, "Disable the GLX_EXT_buffer_age extension") \
   DRI_CONF_DESC(fr, "Disable the GLX_EXT_buffer_age extension") \
   DRI_CONF_DESC(sv, "Disable the GLX_EXT_buffer_age extension") \
DRI_CONF_OPT_END

#define DRI_CONF_DISABLE_OML_SYNC_CONTROL(def) \
DRI_CONF_OPT_BEGIN_B(glx_disable_oml_sync_control, def) \
   DRI_CONF_DESC(en, "Disable the GLX_OML_sync_control extension") \
   DRI_CONF_DESC(ca, "Disable the GLX_OML_sync_control extension") \
   DRI_CONF_DESC(de, "Disable the GLX_OML_sync_control extension") \
   DRI_CONF_DESC(es, "Disable the GLX_OML_sync_control extension") \
   DRI_CONF_DESC(nl, "Disable the GLX_OML_sync_control extension") \
   DRI_CONF_DESC(fr, "Disable the GLX_OML_sync_control extension") \
   DRI_CONF_DESC(sv, "Disable the GLX_OML_sync_control extension") \
DRI_CONF_OPT_END

#define DRI_CONF_DISABLE_SGI_VIDEO_SYNC(def) \
DRI_CONF_OPT_BEGIN_B(glx_disable_sgi_video_sync, def) \
   DRI_CONF_DESC(en, "Disable the GLX_SGI_video_sync extension") \
   DRI_CONF_DESC(ca, "Disable the GLX_SGI_video_sync extension") \
   DRI_CONF_DESC(de, "Disable the GLX_SGI_video_sync extension") \
   DRI_CONF_DESC(es, "Disable the GLX_SGI_video_sync extension") \
   DRI_CONF_DESC(nl, "Disable the GLX_SGI_video_sync extension") \
   DRI_CONF_DESC(fr, "Disable the GLX_SGI_video_sync extension") \
   DRI_CONF_DESC(sv, "Disable the GLX_SGI_video_sync extension") \
DRI_CONF_OPT_END



/**
 * \brief Miscellaneous configuration options
 */
#define DRI_CONF_SECTION_MISCELLANEOUS \
DRI_CONF_SECTION_BEGIN \
        DRI_CONF_DESC(en,"Miscellaneous") \
        DRI_CONF_DESC(ca,"Miscel·lània") \
        DRI_CONF_DESC(de,"Miscellaneous") \
        DRI_CONF_DESC(es,"Misceláneos") \
        DRI_CONF_DESC(nl,"Miscellaneous") \
        DRI_CONF_DESC(fr,"Miscellaneous") \
        DRI_CONF_DESC(sv,"Miscellaneous")

#define DRI_CONF_ALWAYS_HAVE_DEPTH_BUFFER(def) \
DRI_CONF_OPT_BEGIN_B(always_have_depth_buffer, def) \
        DRI_CONF_DESC(en,"Create all visuals with a depth buffer") \
        DRI_CONF_DESC(ca,"Crea tots els visuals amb buffer de profunditat") \
        DRI_CONF_DESC(de,"Create all visuals with a depth buffer") \
        DRI_CONF_DESC(es,"Crear todos los visuales con búfer de profundidad") \
        DRI_CONF_DESC(nl,"Create all visuals with a depth buffer") \
        DRI_CONF_DESC(fr,"Create all visuals with a depth buffer") \
        DRI_CONF_DESC(sv,"Create all visuals with a depth buffer") \
DRI_CONF_OPT_END

#define DRI_CONF_GLSL_ZERO_INIT(def) \
DRI_CONF_OPT_BEGIN_B(glsl_zero_init, def) \
        DRI_CONF_DESC(en,"Force uninitialized variables to default to zero") \
        DRI_CONF_DESC(ca,"Force uninitialized variables to default to zero") \
        DRI_CONF_DESC(de,"Force uninitialized variables to default to zero") \
        DRI_CONF_DESC(es,"Force uninitialized variables to default to zero") \
        DRI_CONF_DESC(nl,"Force uninitialized variables to default to zero") \
        DRI_CONF_DESC(fr,"Force uninitialized variables to default to zero") \
        DRI_CONF_DESC(sv,"Force uninitialized variables to default to zero") \
DRI_CONF_OPT_END

#define DRI_CONF_ALLOW_RGB10_CONFIGS(def) \
DRI_CONF_OPT_BEGIN_B(allow_rgb10_configs, def) \
DRI_CONF_DESC(en,"Allow exposure of visuals and fbconfigs with rgb10a2 formats") \
DRI_CONF_DESC(ca,"Allow exposure of visuals and fbconfigs with rgb10a2 formats") \
DRI_CONF_DESC(de,"Allow exposure of visuals and fbconfigs with rgb10a2 formats") \
DRI_CONF_DESC(es,"Allow exposure of visuals and fbconfigs with rgb10a2 formats") \
DRI_CONF_DESC(nl,"Allow exposure of visuals and fbconfigs with rgb10a2 formats") \
DRI_CONF_DESC(fr,"Allow exposure of visuals and fbconfigs with rgb10a2 formats") \
DRI_CONF_DESC(sv,"Allow exposure of visuals and fbconfigs with rgb10a2 formats") \
DRI_CONF_OPT_END

/**
 * \brief Initialization configuration options
 */
#define DRI_CONF_SECTION_INITIALIZATION \
DRI_CONF_SECTION_BEGIN \
        DRI_CONF_DESC(en,"Initialization") \
        DRI_CONF_DESC(ca,"Inicialització") \
        DRI_CONF_DESC(de,"Initialization") \
        DRI_CONF_DESC(es,"Inicialización") \
        DRI_CONF_DESC(nl,"Initialization") \
        DRI_CONF_DESC(fr,"Initialization") \
        DRI_CONF_DESC(sv,"Initialization")

#define DRI_CONF_DEVICE_ID_PATH_TAG(def) \
DRI_CONF_OPT_BEGIN(device_id, string, def) \
        DRI_CONF_DESC(en,"Define the graphic device to use if possible") \
        DRI_CONF_DESC(ca,"Defineix el dispositiu de gràfics que utilitzar si és possible") \
        DRI_CONF_DESC(de,"Define the graphic device to use if possible") \
        DRI_CONF_DESC(es,"Define el dispositivo de gráficos que usar si es posible") \
        DRI_CONF_DESC(nl,"Define the graphic device to use if possible") \
        DRI_CONF_DESC(fr,"Define the graphic device to use if possible") \
        DRI_CONF_DESC(sv,"Define the graphic device to use if possible") \
DRI_CONF_OPT_END

#define DRI_CONF_DRI_DRIVER(def) \
DRI_CONF_OPT_BEGIN(dri_driver, string, def) \
        DRI_CONF_DESC(en,"Override the DRI driver to load") \
        DRI_CONF_DESC(ca,"Override the DRI driver to load") \
        DRI_CONF_DESC(de,"Override the DRI driver to load") \
        DRI_CONF_DESC(es,"Override the DRI driver to load") \
        DRI_CONF_DESC(nl,"Override the DRI driver to load") \
        DRI_CONF_DESC(fr,"Override the DRI driver to load") \
        DRI_CONF_DESC(sv,"Override the DRI driver to load") \
DRI_CONF_OPT_END

/**
 * \brief Gallium-Nine specific configuration options
 */

#define DRI_CONF_SECTION_NINE \
DRI_CONF_SECTION_BEGIN \
        DRI_CONF_DESC(en,"Gallium Nine") \
        DRI_CONF_DESC(ca,"Gallium Nine") \
        DRI_CONF_DESC(de,"Gallium Nine") \
        DRI_CONF_DESC(es,"Gallium Nine") \
        DRI_CONF_DESC(nl,"Gallium Nine") \
        DRI_CONF_DESC(fr,"Gallium Nine") \
        DRI_CONF_DESC(sv,"Gallium Nine")

#define DRI_CONF_NINE_THROTTLE(def) \
DRI_CONF_OPT_BEGIN(throttle_value, int, def) \
        DRI_CONF_DESC(en,"Define the throttling value. -1 for no throttling, -2 for default (usually 2), 0 for glfinish behaviour") \
        DRI_CONF_DESC(ca,"Defineix el valor de regulació. -1 per a no regular, -2 per al predeterminat (generalment 2), 0 per al comportament de glfinish") \
        DRI_CONF_DESC(de,"Define the throttling value. -1 for no throttling, -2 for default (usually 2), 0 for glfinish behaviour") \
        DRI_CONF_DESC(es,"Define el valor de regulación. -1 para no regular, -2 para el por defecto (generalmente 2), 0 para el comportamiento de glfinish") \
        DRI_CONF_DESC(nl,"Define the throttling value. -1 for no throttling, -2 for default (usually 2), 0 for glfinish behaviour") \
        DRI_CONF_DESC(fr,"Define the throttling value. -1 for no throttling, -2 for default (usually 2), 0 for glfinish behaviour") \
        DRI_CONF_DESC(sv,"Define the throttling value. -1 for no throttling, -2 for default (usually 2), 0 for glfinish behaviour") \
DRI_CONF_OPT_END

#define DRI_CONF_NINE_THREADSUBMIT(def) \
DRI_CONF_OPT_BEGIN_B(thread_submit, def) \
        DRI_CONF_DESC(en,"Use an additional thread to submit buffers.") \
        DRI_CONF_DESC(ca,"Utilitza un fil addicional per a entregar els buffers.") \
        DRI_CONF_DESC(de,"Use an additional thread to submit buffers.") \
        DRI_CONF_DESC(es,"Usar un hilo adicional para entregar los búfer.") \
        DRI_CONF_DESC(nl,"Use an additional thread to submit buffers.") \
        DRI_CONF_DESC(fr,"Use an additional thread to submit buffers.") \
        DRI_CONF_DESC(sv,"Use an additional thread to submit buffers.") \
DRI_CONF_OPT_END

#define DRI_CONF_NINE_OVERRIDEVENDOR(def) \
DRI_CONF_OPT_BEGIN(override_vendorid, int, def) \
        DRI_CONF_DESC(en,"Define the vendor_id to report. This allows faking another hardware vendor.") \
        DRI_CONF_DESC(ca,"Define the vendor_id to report. This allows faking another hardware vendor.") \
        DRI_CONF_DESC(de,"Define the vendor_id to report. This allows faking another hardware vendor.") \
        DRI_CONF_DESC(es,"Define the vendor_id to report. This allows faking another hardware vendor.") \
        DRI_CONF_DESC(nl,"Define the vendor_id to report. This allows faking another hardware vendor.") \
        DRI_CONF_DESC(fr,"Define the vendor_id to report. This allows faking another hardware vendor.") \
        DRI_CONF_DESC(sv,"Define the vendor_id to report. This allows faking another hardware vendor.") \
DRI_CONF_OPT_END

#define DRI_CONF_NINE_ALLOWDISCARDDELAYEDRELEASE(def) \
DRI_CONF_OPT_BEGIN_B(discard_delayed_release, def) \
        DRI_CONF_DESC(en,"Whether to allow the display server to release buffers with a delay when using d3d's presentation mode DISCARD. Default to true. Set to false if suffering from lag (thread_submit=true can also help in this situation).") \
        DRI_CONF_DESC(ca,"Whether to allow the display server to release buffers with a delay when using d3d's presentation mode DISCARD. Default to true. Set to false if suffering from lag (thread_submit=true can also help in this situation).") \
        DRI_CONF_DESC(de,"Whether to allow the display server to release buffers with a delay when using d3d's presentation mode DISCARD. Default to true. Set to false if suffering from lag (thread_submit=true can also help in this situation).") \
        DRI_CONF_DESC(es,"Whether to allow the display server to release buffers with a delay when using d3d's presentation mode DISCARD. Default to true. Set to false if suffering from lag (thread_submit=true can also help in this situation).") \
        DRI_CONF_DESC(nl,"Whether to allow the display server to release buffers with a delay when using d3d's presentation mode DISCARD. Default to true. Set to false if suffering from lag (thread_submit=true can also help in this situation).") \
        DRI_CONF_DESC(fr,"Whether to allow the display server to release buffers with a delay when using d3d's presentation mode DISCARD. Default to true. Set to false if suffering from lag (thread_submit=true can also help in this situation).") \
        DRI_CONF_DESC(sv,"Whether to allow the display server to release buffers with a delay when using d3d's presentation mode DISCARD. Default to true. Set to false if suffering from lag (thread_submit=true can also help in this situation).") \
DRI_CONF_OPT_END

#define DRI_CONF_NINE_TEARFREEDISCARD(def) \
DRI_CONF_OPT_BEGIN_B(tearfree_discard, def) \
        DRI_CONF_DESC(en,"Whether to make d3d's presentation mode DISCARD (games usually use that mode) Tear Free. If rendering above screen refresh, some frames will get skipped. false by default.") \
        DRI_CONF_DESC(ca,"Whether to make d3d's presentation mode DISCARD (games usually use that mode) Tear Free. If rendering above screen refresh, some frames will get skipped. false by default.") \
        DRI_CONF_DESC(de,"Whether to make d3d's presentation mode DISCARD (games usually use that mode) Tear Free. If rendering above screen refresh, some frames will get skipped. false by default.") \
        DRI_CONF_DESC(es,"Whether to make d3d's presentation mode DISCARD (games usually use that mode) Tear Free. If rendering above screen refresh, some frames will get skipped. false by default.") \
        DRI_CONF_DESC(nl,"Whether to make d3d's presentation mode DISCARD (games usually use that mode) Tear Free. If rendering above screen refresh, some frames will get skipped. false by default.") \
        DRI_CONF_DESC(fr,"Whether to make d3d's presentation mode DISCARD (games usually use that mode) Tear Free. If rendering above screen refresh, some frames will get skipped. false by default.") \
        DRI_CONF_DESC(sv,"Whether to make d3d's presentation mode DISCARD (games usually use that mode) Tear Free. If rendering above screen refresh, some frames will get skipped. false by default.") \
DRI_CONF_OPT_END

#define DRI_CONF_NINE_CSMT(def) \
DRI_CONF_OPT_BEGIN(csmt_force, int, def) \
        DRI_CONF_DESC(en,"If set to 1, force gallium nine CSMT. If set to 0, disable it. By default (-1) CSMT is enabled on known thread-safe drivers.") \
        DRI_CONF_DESC(ca,"If set to 1, force gallium nine CSMT. If set to 0, disable it. By default (-1) CSMT is enabled on known thread-safe drivers.") \
        DRI_CONF_DESC(de,"If set to 1, force gallium nine CSMT. If set to 0, disable it. By default (-1) CSMT is enabled on known thread-safe drivers.") \
        DRI_CONF_DESC(es,"If set to 1, force gallium nine CSMT. If set to 0, disable it. By default (-1) CSMT is enabled on known thread-safe drivers.") \
        DRI_CONF_DESC(nl,"If set to 1, force gallium nine CSMT. If set to 0, disable it. By default (-1) CSMT is enabled on known thread-safe drivers.") \
        DRI_CONF_DESC(fr,"If set to 1, force gallium nine CSMT. If set to 0, disable it. By default (-1) CSMT is enabled on known thread-safe drivers.") \
        DRI_CONF_DESC(sv,"If set to 1, force gallium nine CSMT. If set to 0, disable it. By default (-1) CSMT is enabled on known thread-safe drivers.") \
DRI_CONF_OPT_END

/**
 * \brief radeonsi specific configuration options
 */

#define DRI_CONF_RADEONSI_ENABLE_SISCHED(def) \
DRI_CONF_OPT_BEGIN_B(radeonsi_enable_sisched, def) \
        DRI_CONF_DESC(en,"Use the LLVM sisched option for shader compiles") \
        DRI_CONF_DESC(ca,"Use the LLVM sisched option for shader compiles") \
        DRI_CONF_DESC(de,"Use the LLVM sisched option for shader compiles") \
        DRI_CONF_DESC(es,"Use the LLVM sisched option for shader compiles") \
        DRI_CONF_DESC(nl,"Use the LLVM sisched option for shader compiles") \
        DRI_CONF_DESC(fr,"Use the LLVM sisched option for shader compiles") \
        DRI_CONF_DESC(sv,"Use the LLVM sisched option for shader compiles") \
DRI_CONF_OPT_END

#define DRI_CONF_RADEONSI_ASSUME_NO_Z_FIGHTS(def) \
DRI_CONF_OPT_BEGIN_B(radeonsi_assume_no_z_fights, def) \
        DRI_CONF_DESC(en,"Assume no Z fights (enables aggressive out-of-order rasterization to improve performance; may cause rendering errors)") \
        DRI_CONF_DESC(ca,"Assume no Z fights (enables aggressive out-of-order rasterization to improve performance; may cause rendering errors)") \
        DRI_CONF_DESC(de,"Assume no Z fights (enables aggressive out-of-order rasterization to improve performance; may cause rendering errors)") \
        DRI_CONF_DESC(es,"Assume no Z fights (enables aggressive out-of-order rasterization to improve performance; may cause rendering errors)") \
        DRI_CONF_DESC(nl,"Assume no Z fights (enables aggressive out-of-order rasterization to improve performance; may cause rendering errors)") \
        DRI_CONF_DESC(fr,"Assume no Z fights (enables aggressive out-of-order rasterization to improve performance; may cause rendering errors)") \
        DRI_CONF_DESC(sv,"Assume no Z fights (enables aggressive out-of-order rasterization to improve performance; may cause rendering errors)") \
DRI_CONF_OPT_END

#define DRI_CONF_RADEONSI_COMMUTATIVE_BLEND_ADD(def) \
DRI_CONF_OPT_BEGIN_B(radeonsi_commutative_blend_add, def) \
        DRI_CONF_DESC(en,"Commutative additive blending optimizations (may cause rendering errors)") \
        DRI_CONF_DESC(ca,"Commutative additive blending optimizations (may cause rendering errors)") \
        DRI_CONF_DESC(de,"Commutative additive blending optimizations (may cause rendering errors)") \
        DRI_CONF_DESC(es,"Commutative additive blending optimizations (may cause rendering errors)") \
        DRI_CONF_DESC(nl,"Commutative additive blending optimizations (may cause rendering errors)") \
        DRI_CONF_DESC(fr,"Commutative additive blending optimizations (may cause rendering errors)") \
        DRI_CONF_DESC(sv,"Commutative additive blending optimizations (may cause rendering errors)") \
DRI_CONF_OPT_END

#define DRI_CONF_RADEONSI_CLEAR_DB_CACHE_BEFORE_CLEAR(def) \
DRI_CONF_OPT_BEGIN_B(radeonsi_clear_db_cache_before_clear, def) \
        DRI_CONF_DESC(en,"Clear DB cache before fast depth clear") \
DRI_CONF_OPT_END

#define DRI_CONF_RADEONSI_ZERO_ALL_VRAM_ALLOCS(def) \
DRI_CONF_OPT_BEGIN_B(radeonsi_zerovram, def) \
        DRI_CONF_DESC(en,"Zero all vram allocations") \
DRI_CONF_OPT_END
