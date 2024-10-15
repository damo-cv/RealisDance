/*
 * Copyright (c) 2015 Intel Corporation
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

#include <stdlib.h>
#include <math.h>

#include "util/macros.h"
#include "main/macros.h"

#include "gen_l3_config.h"

/**
 * IVB/HSW validated L3 configurations.  The first entry will be used as
 * default by gen7_restore_default_l3_config(), otherwise the ordering is
 * unimportant.
 */
static const struct gen_l3_config ivb_l3_configs[] = {
   /* SLM URB ALL DC  RO  IS   C   T */
   {{  0, 32,  0,  0, 32,  0,  0,  0 }},
   {{  0, 32,  0, 16, 16,  0,  0,  0 }},
   {{  0, 32,  0,  4,  0,  8,  4, 16 }},
   {{  0, 28,  0,  8,  0,  8,  4, 16 }},
   {{  0, 28,  0, 16,  0,  8,  4,  8 }},
   {{  0, 28,  0,  8,  0, 16,  4,  8 }},
   {{  0, 28,  0,  0,  0, 16,  4, 16 }},
   {{  0, 32,  0,  0,  0, 16,  0, 16 }},
   {{  0, 28,  0,  4, 32,  0,  0,  0 }},
   {{ 16, 16,  0, 16, 16,  0,  0,  0 }},
   {{ 16, 16,  0,  8,  0,  8,  8,  8 }},
   {{ 16, 16,  0,  4,  0,  8,  4, 16 }},
   {{ 16, 16,  0,  4,  0, 16,  4,  8 }},
   {{ 16, 16,  0,  0, 32,  0,  0,  0 }},
   {{ 0 }}
};

/**
 * VLV validated L3 configurations.  \sa ivb_l3_configs.
 */
static const struct gen_l3_config vlv_l3_configs[] = {
   /* SLM URB ALL DC  RO  IS   C   T */
   {{  0, 64,  0,  0, 32,  0,  0,  0 }},
   {{  0, 80,  0,  0, 16,  0,  0,  0 }},
   {{  0, 80,  0,  8,  8,  0,  0,  0 }},
   {{  0, 64,  0, 16, 16,  0,  0,  0 }},
   {{  0, 60,  0,  4, 32,  0,  0,  0 }},
   {{ 32, 32,  0, 16, 16,  0,  0,  0 }},
   {{ 32, 40,  0,  8, 16,  0,  0,  0 }},
   {{ 32, 40,  0, 16,  8,  0,  0,  0 }},
   {{ 0 }}
};

/**
 * BDW validated L3 configurations.  \sa ivb_l3_configs.
 */
static const struct gen_l3_config bdw_l3_configs[] = {
   /* SLM URB ALL DC  RO  IS   C   T */
   {{  0, 48, 48,  0,  0,  0,  0,  0 }},
   {{  0, 48,  0, 16, 32,  0,  0,  0 }},
   {{  0, 32,  0, 16, 48,  0,  0,  0 }},
   {{  0, 32,  0,  0, 64,  0,  0,  0 }},
   {{  0, 32, 64,  0,  0,  0,  0,  0 }},
   {{ 24, 16, 48,  0,  0,  0,  0,  0 }},
   {{ 24, 16,  0, 16, 32,  0,  0,  0 }},
   {{ 24, 16,  0, 32, 16,  0,  0,  0 }},
   {{ 0 }}
};

/**
 * CHV/SKL validated L3 configurations.  \sa ivb_l3_configs.
 */
static const struct gen_l3_config chv_l3_configs[] = {
   /* SLM URB ALL DC  RO  IS   C   T */
   {{  0, 48, 48,  0,  0,  0,  0,  0 }},
   {{  0, 48,  0, 16, 32,  0,  0,  0 }},
   {{  0, 32,  0, 16, 48,  0,  0,  0 }},
   {{  0, 32,  0,  0, 64,  0,  0,  0 }},
   {{  0, 32, 64,  0,  0,  0,  0,  0 }},
   {{ 32, 16, 48,  0,  0,  0,  0,  0 }},
   {{ 32, 16,  0, 16, 32,  0,  0,  0 }},
   {{ 32, 16,  0, 32, 16,  0,  0,  0 }},
   {{ 0 }}
};

/**
 * BXT 2x6 validated L3 configurations.  \sa ivb_l3_configs.
 */
static const struct gen_l3_config bxt_2x6_l3_configs[] = {
   /* SLM URB ALL DC  RO  IS   C   T */
   {{  0, 32, 48,  0,  0,  0,  0,  0 }},
   {{  0, 32,  0,  8, 40,  0,  0,  0 }},
   {{  0, 32,  0, 32, 16,  0,  0,  0 }},
   {{ 16, 16, 48,  0,  0,  0,  0,  0 }},
   {{ 16, 16,  0, 40,  8,  0,  0,  0 }},
   {{ 16, 16,  0, 16, 32,  0,  0,  0 }},
   {{ 0 }}
};

/**
 * CNL validated L3 configurations.  \sa ivb_l3_configs.
 */
static const struct gen_l3_config cnl_l3_configs[] = {
   /* SLM URB ALL DC  RO  IS   C   T */
   {{  0, 64, 64,  0,  0,  0,  0,  0 }},
   {{  0, 64,  0, 16, 48,  0,  0,  0 }},
   {{  0, 48,  0, 16, 64,  0,  0,  0 }},
   {{  0, 32,  0,  0, 96,  0,  0,  0 }},
   {{  0, 32, 96,  0,  0,  0,  0,  0 }},
   {{  0, 32,  0, 16, 80,  0,  0,  0 }},
   {{ 32, 16, 80,  0,  0,  0,  0,  0 }},
   {{ 32, 16,  0, 64, 16,  0,  0,  0 }},
   {{ 32,  0, 96,  0,  0,  0,  0,  0 }},
   {{ 0 }}
};

/**
 * ICL validated L3 configurations.  \sa icl_l3_configs.
 */
static const struct gen_l3_config icl_l3_configs[] = {
   /* SLM URB ALL DC  RO  IS   C   T */
   {{  0, 64, 64,  0,  0,  0,  0,  0 }},
   {{  0, 64,  0, 16, 48,  0,  0,  0 }},
   {{  0, 48,  0, 16, 64,  0,  0,  0 }},
   {{  0, 32,  0,  0, 96,  0,  0,  0 }},
   {{  0, 32, 96,  0,  0,  0,  0,  0 }},
   {{  0, 32,  0, 16, 80,  0,  0,  0 }},
   {{  0 }}
};


/**
 * Return a zero-terminated array of validated L3 configurations for the
 * specified device.
 */
static const struct gen_l3_config *
get_l3_configs(const struct gen_device_info *devinfo)
{
   switch (devinfo->gen) {
   case 7:
      return (devinfo->is_baytrail ? vlv_l3_configs : ivb_l3_configs);

   case 8:
      return (devinfo->is_cherryview ? chv_l3_configs : bdw_l3_configs);

   case 9:
      if (devinfo->l3_banks == 1)
         return bxt_2x6_l3_configs;
      return chv_l3_configs;

   case 10:
      return cnl_l3_configs;

   case 11:
      return icl_l3_configs;

   default:
      unreachable("Not implemented");
   }
}

/**
 * L1-normalize a vector of L3 partition weights.
 */
static struct gen_l3_weights
norm_l3_weights(struct gen_l3_weights w)
{
   float sz = 0;

   for (unsigned i = 0; i < GEN_NUM_L3P; i++)
      sz += w.w[i];

   for (unsigned i = 0; i < GEN_NUM_L3P; i++)
      w.w[i] /= sz;

   return w;
}

/**
 * Get the relative partition weights of the specified L3 configuration.
 */
struct gen_l3_weights
gen_get_l3_config_weights(const struct gen_l3_config *cfg)
{
   if (cfg) {
      struct gen_l3_weights w;

      for (unsigned i = 0; i < GEN_NUM_L3P; i++)
         w.w[i] = cfg->n[i];

      return norm_l3_weights(w);
   } else {
      const struct gen_l3_weights w = { { 0 } };
      return w;
   }
}

/**
 * Distance between two L3 configurations represented as vectors of weights.
 * Usually just the L1 metric except when the two configurations are
 * considered incompatible in which case the distance will be infinite.  Note
 * that the compatibility condition is asymmetric -- They will be considered
 * incompatible whenever the reference configuration \p w0 requires SLM, DC,
 * or URB but \p w1 doesn't provide it.
 */
float
gen_diff_l3_weights(struct gen_l3_weights w0, struct gen_l3_weights w1)
{
   if ((w0.w[GEN_L3P_SLM] && !w1.w[GEN_L3P_SLM]) ||
       (w0.w[GEN_L3P_DC] && !w1.w[GEN_L3P_DC] && !w1.w[GEN_L3P_ALL]) ||
       (w0.w[GEN_L3P_URB] && !w1.w[GEN_L3P_URB])) {
      return HUGE_VALF;

   } else {
      float dw = 0;

      for (unsigned i = 0; i < GEN_NUM_L3P; i++)
         dw += fabs(w0.w[i] - w1.w[i]);

      return dw;
   }
}

/**
 * Return a reasonable default L3 configuration for the specified device based
 * on whether SLM and DC are required.  In the non-SLM non-DC case the result
 * is intended to approximately resemble the hardware defaults.
 */
struct gen_l3_weights
gen_get_default_l3_weights(const struct gen_device_info *devinfo,
                           bool needs_dc, bool needs_slm)
{
   struct gen_l3_weights w = {{ 0 }};

   w.w[GEN_L3P_SLM] = devinfo->gen < 11 && needs_slm;
   w.w[GEN_L3P_URB] = 1.0;

   if (devinfo->gen >= 8) {
      w.w[GEN_L3P_ALL] = 1.0;
   } else {
      w.w[GEN_L3P_DC] = needs_dc ? 0.1 : 0;
      w.w[GEN_L3P_RO] = devinfo->is_baytrail ? 0.5 : 1.0;
   }

   return norm_l3_weights(w);
}

/**
 * Get the default L3 configuration
 */
const struct gen_l3_config *
gen_get_default_l3_config(const struct gen_device_info *devinfo)
{
   /* For efficiency assume that the first entry of the array matches the
    * default configuration.
    */
   const struct gen_l3_config *const cfg = get_l3_configs(devinfo);
   assert(cfg == gen_get_l3_config(devinfo,
                    gen_get_default_l3_weights(devinfo, false, false)));
   return cfg;
}

/**
 * Return the closest validated L3 configuration for the specified device and
 * weight vector.
 */
const struct gen_l3_config *
gen_get_l3_config(const struct gen_device_info *devinfo,
                  struct gen_l3_weights w0)
{
   const struct gen_l3_config *const cfgs = get_l3_configs(devinfo);
   const struct gen_l3_config *cfg_best = NULL;
   float dw_best = HUGE_VALF;

   for (const struct gen_l3_config *cfg = cfgs; cfg->n[GEN_L3P_URB]; cfg++) {
      const float dw = gen_diff_l3_weights(w0, gen_get_l3_config_weights(cfg));

      if (dw < dw_best) {
         cfg_best = cfg;
         dw_best = dw;
      }
   }

   return cfg_best;
}

/**
 * Return the size of an L3 way in KB.
 */
static unsigned
get_l3_way_size(const struct gen_device_info *devinfo)
{
   const unsigned way_size_per_bank =
      devinfo->gen >= 9 && devinfo->l3_banks == 1 ? 4 : 2;

   assert(devinfo->l3_banks);
   return way_size_per_bank * devinfo->l3_banks;
}

/**
 * Return the unit brw_context::urb::size is expressed in, in KB.  \sa
 * gen_device_info::urb::size.
 */
static unsigned
get_urb_size_scale(const struct gen_device_info *devinfo)
{
   return (devinfo->gen >= 8 ? devinfo->num_slices : 1);
}

unsigned
gen_get_l3_config_urb_size(const struct gen_device_info *devinfo,
                           const struct gen_l3_config *cfg)
{
   /* From the SKL "L3 Allocation and Programming" documentation:
    *
    * "URB is limited to 1008KB due to programming restrictions.  This is not
    * a restriction of the L3 implementation, but of the FF and other clients.
    * Therefore, in a GT4 implementation it is possible for the programmed
    * allocation of the L3 data array to provide 3*384KB=1152KB for URB, but
    * only 1008KB of this will be used."
    */
   const unsigned max = (devinfo->gen == 9 ? 1008 : ~0);
   return MIN2(max, cfg->n[GEN_L3P_URB] * get_l3_way_size(devinfo)) /
          get_urb_size_scale(devinfo);
}

/**
 * Print out the specified L3 configuration.
 */
void
gen_dump_l3_config(const struct gen_l3_config *cfg, FILE *fp)
{
   fprintf(stderr, "SLM=%d URB=%d ALL=%d DC=%d RO=%d IS=%d C=%d T=%d\n",
           cfg->n[GEN_L3P_SLM], cfg->n[GEN_L3P_URB], cfg->n[GEN_L3P_ALL],
           cfg->n[GEN_L3P_DC], cfg->n[GEN_L3P_RO],
           cfg->n[GEN_L3P_IS], cfg->n[GEN_L3P_C], cfg->n[GEN_L3P_T]);
}
