/*
 * Copyright © 2014-2017 Broadcom
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

/**
 * @file v3d_simulator_hw.c
 *
 * Implements the actual HW interaction betweeh the GL driver's VC5 simulator and the simulator.
 *
 * The register headers between V3D versions will have conflicting defines, so
 * all register interactions appear in this file and are compiled per V3D version
 * we support.
 */

#ifdef USE_V3D_SIMULATOR

#include "v3d_screen.h"
#include "v3d_context.h"
#include "v3d_simulator_wrapper.h"

#define HW_REGISTER_RO(x) (x)
#define HW_REGISTER_RW(x) (x)
#if V3D_VERSION >= 41
#include "libs/core/v3d/registers/4.1.34.0/v3d.h"
#else
#include "libs/core/v3d/registers/3.3.0.0/v3d.h"
#endif

#define V3D_WRITE(reg, val) v3d_hw_write_reg(v3d, reg, val)
#define V3D_READ(reg) v3d_hw_read_reg(v3d, reg)

static void
v3d_flush_l3(struct v3d_hw *v3d)
{
        if (!v3d_hw_has_gca(v3d))
                return;

#if V3D_VERSION < 40
        uint32_t gca_ctrl = V3D_READ(V3D_GCA_CACHE_CTRL);

        V3D_WRITE(V3D_GCA_CACHE_CTRL, gca_ctrl | V3D_GCA_CACHE_CTRL_FLUSH_SET);
        V3D_WRITE(V3D_GCA_CACHE_CTRL, gca_ctrl & ~V3D_GCA_CACHE_CTRL_FLUSH_SET);
#endif
}

/* Invalidates the L2 cache.  This is a read-only cache. */
static void
v3d_flush_l2(struct v3d_hw *v3d)
{
        V3D_WRITE(V3D_CTL_0_L2CACTL,
                  V3D_CTL_0_L2CACTL_L2CCLR_SET |
                  V3D_CTL_0_L2CACTL_L2CENA_SET);
}

/* Invalidates texture L2 cachelines */
static void
v3d_flush_l2t(struct v3d_hw *v3d)
{
        V3D_WRITE(V3D_CTL_0_L2TFLSTA, 0);
        V3D_WRITE(V3D_CTL_0_L2TFLEND, ~0);
        V3D_WRITE(V3D_CTL_0_L2TCACTL,
                  V3D_CTL_0_L2TCACTL_L2TFLS_SET |
                  (0 << V3D_CTL_0_L2TCACTL_L2TFLM_LSB));
}

/* Invalidates the slice caches.  These are read-only caches. */
static void
v3d_flush_slices(struct v3d_hw *v3d)
{
        V3D_WRITE(V3D_CTL_0_SLCACTL, ~0);
}

static void
v3d_flush_caches(struct v3d_hw *v3d)
{
        v3d_flush_l3(v3d);
        v3d_flush_l2(v3d);
        v3d_flush_l2t(v3d);
        v3d_flush_slices(v3d);
}

int
v3dX(simulator_get_param_ioctl)(struct v3d_hw *v3d,
                                struct drm_v3d_get_param *args)
{
        static const uint32_t reg_map[] = {
                [DRM_V3D_PARAM_V3D_UIFCFG] = V3D_HUB_CTL_UIFCFG,
                [DRM_V3D_PARAM_V3D_HUB_IDENT1] = V3D_HUB_CTL_IDENT1,
                [DRM_V3D_PARAM_V3D_HUB_IDENT2] = V3D_HUB_CTL_IDENT2,
                [DRM_V3D_PARAM_V3D_HUB_IDENT3] = V3D_HUB_CTL_IDENT3,
                [DRM_V3D_PARAM_V3D_CORE0_IDENT0] = V3D_CTL_0_IDENT0,
                [DRM_V3D_PARAM_V3D_CORE0_IDENT1] = V3D_CTL_0_IDENT1,
                [DRM_V3D_PARAM_V3D_CORE0_IDENT2] = V3D_CTL_0_IDENT2,
        };

        if (args->param < ARRAY_SIZE(reg_map) && reg_map[args->param]) {
                args->value = V3D_READ(reg_map[args->param]);
                return 0;
        }

        fprintf(stderr, "Unknown DRM_IOCTL_VC5_GET_PARAM(%lld)\n",
                (long long)args->value);
        abort();
}

void
v3dX(simulator_init_regs)(struct v3d_hw *v3d)
{
#if V3D_VERSION == 33
        /* Set OVRTMUOUT to match kernel behavior.
         *
         * This means that the texture sampler uniform configuration's tmu
         * output type field is used, instead of using the hardware default
         * behavior based on the texture type.  If you want the default
         * behavior, you can still put "2" in the indirect texture state's
         * output_type field.
         */
        V3D_WRITE(V3D_CTL_0_MISCCFG, V3D_CTL_1_MISCCFG_OVRTMUOUT_SET);
#endif
}

void
v3dX(simulator_flush)(struct v3d_hw *v3d, struct drm_v3d_submit_cl *submit,
                      uint32_t gmp_ofs)
{
        /* Completely reset the GMP. */
        V3D_WRITE(V3D_GMP_0_CFG,
                  V3D_GMP_0_CFG_PROTENABLE_SET);
        V3D_WRITE(V3D_GMP_0_TABLE_ADDR, gmp_ofs);
        V3D_WRITE(V3D_GMP_0_CLEAR_LOAD, ~0);
        while (V3D_READ(V3D_GMP_0_STATUS) &
               V3D_GMP_0_STATUS_CFG_BUSY_SET) {
                ;
        }

        v3d_flush_caches(v3d);

        if (submit->qma) {
                V3D_WRITE(V3D_CLE_0_CT0QMA, submit->qma);
                V3D_WRITE(V3D_CLE_0_CT0QMS, submit->qms);
        }
#if V3D_VERSION >= 41
        if (submit->qts) {
                V3D_WRITE(V3D_CLE_0_CT0QTS,
                          V3D_CLE_0_CT0QTS_CTQTSEN_SET |
                          submit->qts);
        }
#endif
        V3D_WRITE(V3D_CLE_0_CT0QBA, submit->bcl_start);
        V3D_WRITE(V3D_CLE_0_CT0QEA, submit->bcl_end);

        /* Wait for bin to complete before firing render, as it seems the
         * simulator doesn't implement the semaphores.
         */
        while (V3D_READ(V3D_CLE_0_CT0CA) !=
               V3D_READ(V3D_CLE_0_CT0EA)) {
                v3d_hw_tick(v3d);
        }

        V3D_WRITE(V3D_CLE_0_CT1QBA, submit->rcl_start);
        V3D_WRITE(V3D_CLE_0_CT1QEA, submit->rcl_end);

        while (V3D_READ(V3D_CLE_0_CT1CA) !=
               V3D_READ(V3D_CLE_0_CT1EA) ||
               V3D_READ(V3D_CLE_1_CT1CA) !=
               V3D_READ(V3D_CLE_1_CT1EA)) {
                v3d_hw_tick(v3d);
        }
}

#endif /* USE_V3D_SIMULATOR */
