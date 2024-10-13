#ifndef _DRM_HELPER_PUBLIC_H
#define _DRM_HELPER_PUBLIC_H

enum drm_conf;
struct drm_conf_ret;

struct pipe_screen;
struct pipe_screen_config;

struct pipe_screen *
pipe_i915_create_screen(int fd, const struct pipe_screen_config *config);

struct pipe_screen *
pipe_ilo_create_screen(int fd, const struct pipe_screen_config *config);

struct pipe_screen *
pipe_nouveau_create_screen(int fd, const struct pipe_screen_config *config);

struct pipe_screen *
pipe_r300_create_screen(int fd, const struct pipe_screen_config *config);

struct pipe_screen *
pipe_r600_create_screen(int fd, const struct pipe_screen_config *config);

struct pipe_screen *
pipe_radeonsi_create_screen(int fd, const struct pipe_screen_config *config);
const struct drm_conf_ret *
pipe_radeonsi_configuration_query(enum drm_conf conf);

struct pipe_screen *
pipe_vmwgfx_create_screen(int fd, const struct pipe_screen_config *config);

struct pipe_screen *
pipe_freedreno_create_screen(int fd, const struct pipe_screen_config *config);

struct pipe_screen *
pipe_virgl_create_screen(int fd, const struct pipe_screen_config *config);

struct pipe_screen *
pipe_v3d_create_screen(int fd, const struct pipe_screen_config *config);

struct pipe_screen *
pipe_vc4_create_screen(int fd, const struct pipe_screen_config *config);

struct pipe_screen *
pipe_pl111_create_screen(int fd, const struct pipe_screen_config *config);

struct pipe_screen *
pipe_etna_create_screen(int fd, const struct pipe_screen_config *config);

struct pipe_screen *
pipe_imx_drm_create_screen(int fd, const struct pipe_screen_config *config);

struct pipe_screen *
pipe_tegra_create_screen(int fd, const struct pipe_screen_config *config);

const struct drm_conf_ret *
pipe_default_configuration_query(enum drm_conf conf);

#endif /* _DRM_HELPER_PUBLIC_H */
