# Utilities API

Frequently used math, visualization, and export helpers pulled from their docstrings. Diffusion summaries and NIfTI exporters live in `prism.dmri.utils`, spherical harmonics and plots in `prism.dmri.shm` and `prism.dmri.viz`. `prism.utils.transform` holds generic coordinate transforms.

## Diffusion MRI helpers

::: prism.dmri.utils
    options:
      show_root_heading: true
      show_root_full_path: false
      filters:
        - "^ssfp_signal_fn$"
        - "^freed_ssfp_signal_fn$"
        - "^fit_diffusion_tensor_linearized$"
        - "^cart2sph$"
        - "^sph2cart$"
        - "^normalize_bvecs$"
        - "^compute_fa$"
        - "^compute_md$"
        - "^compute_rd$"
        - "^compute_ad$"
        - "^rotation_matrix_100_to_theta_phi$"
        - "^rotation_matrix_100_to_xyz$"
        - "^rotation_matrix_100_to_theta_phi_psi$"
        - "^rotation_matrix_around_100$"
        - "^canonical_bingham_normalization_series$"
        - "^make_dyads$"
        - "^export_nifti$"
        - "^export_sbi_estimates$"

## Transforms and math

::: prism.utils.transform
    options:
      show_root_heading: true
      show_root_full_path: false

::: prism.dmri.shm
    options:
      show_root_heading: true
      show_root_full_path: false

::: prism.dmri.viz
    options:
      show_root_heading: true
      show_root_full_path: false
