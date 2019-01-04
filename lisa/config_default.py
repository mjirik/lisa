#! /usr/bin/python
# -*- coding: utf-8 -*-

import logging
logger = logging.getLogger(__name__)

default_segmentation_alternative_params = {
        "simple 1 mm":{
            "segmodelparams": {
                'mdl_stored_file': None,
                'fv_type': 'intensity',
                'fv_extern': None,
                # 'mdl_stored_file': '~/lisa_data/liver_intensity.Model.p',
                # "fv_type": 'fv_extern',
                # 'fv_extern':'intensity_localization_fv',
                # 'method': 'multiscale_graphcut',
                "params": {"covariance_type": "full", "n_components": 3}
            },
            'working_voxelsize_mm': 1
        },
    "simple 1.3 mm":{
        "segmodelparams": {
            'mdl_stored_file': None,
            'fv_type': 'intensity',
            'fv_extern': None,
            # 'mdl_stored_file': '~/lisa_data/liver_intensity.Model.p',
            # "fv_type": 'fv_extern',
            # 'fv_extern':'intensity_localization_fv',
            # 'method': 'multiscale_graphcut',
            "params": {"covariance_type": "full", "n_components": 3}
        },
        'working_voxelsize_mm': 1.3
    },
        "simple 1.5 mm":{
            "segmodelparams": {
                'mdl_stored_file': None,
                'fv_type': 'intensity',
                'fv_extern': None,
                # 'mdl_stored_file': '~/lisa_data/liver_intensity.Model.p',
                # "fv_type": 'fv_extern',
                # 'fv_extern':'intensity_localization_fv',
                # 'method': 'multiscale_graphcut',
                "params": {"covariance_type": "full", "n_components": 3}
            },
            'working_voxelsize_mm': 1.5
        },
        "simple 2 mm":{
            "segmodelparams": {
                'mdl_stored_file': None,
                'fv_type': 'intensity',
                'fv_extern': None,
                # 'mdl_stored_file': '~/lisa_data/liver_intensity.Model.p',
                # "fv_type": 'fv_extern',
                # 'fv_extern':'intensity_localization_fv',
                # 'method': 'multiscale_graphcut',
                "params": {"covariance_type": "full", "n_components": 3}
            },
            'working_voxelsize_mm': 2
        },
        "simple 2.5 mm":{
            "segmodelparams": {
                'mdl_stored_file': None,
                'fv_type': 'intensity',
                'fv_extern': None,
                # 'mdl_stored_file': '~/lisa_data/liver_intensity.Model.p',
                # "fv_type": 'fv_extern',
                # 'fv_extern':'intensity_localization_fv',
                # 'method': 'multiscale_graphcut',
                "params": {"covariance_type": "full", "n_components": 3}
            },
            'working_voxelsize_mm': 2.5
        },
        "label left kidney":{
            "output_label": 'left kidney',
            'clean_seeds_after_update_parameters': True,
        },
        "label right kidney":{
            "output_label": 'right kidney',
            'clean_seeds_after_update_parameters': True,
        },
        "label liver":{
            "output_label": 'liver',
            'clean_seeds_after_update_parameters': True,
        },
        "label hearth":{
            "output_label": 'hearth',
            'clean_seeds_after_update_parameters': True,
        },
        "msgc_lo2hi": {
            "segparams": {
                'method':'multiscale_graphcut_lo2hi',
                'use_boundary_penalties': True,
                'boundary_dilatation_distance': 1,
                'boundary_penalties_weight': 1,
                'block_size': 8,
                'tile_zoom_constant': 1,
                "pairwise_alpha_per_mm2":45,
                "return_only_object_with_seeds": True
            }
        },
        "graphcut": {
            "segparams": {
                'method':'graphcut',
                'use_boundary_penalties': True,
                'boundary_dilatation_distance': 2,
                'boundary_penalties_weight': 1,
                'block_size': 10,
                'tile_zoom_constant': 1,
                "pairwise_alpha_per_mm2":45,
                "return_only_object_with_seeds": True
            }
        }

    }

CONFIG_DEFAULT={
    "autocrop": False,
    "autocrop_margin_mm": [10, 10, 10],
    "data3d": None,
    "datapath": None,
    "edit_data": False,
    "manualroi": True,
    "metadata": None,
    "output_label": 1,
    "qt_app": None,
    "roi": None,
    "seeds": None,
    "segmentation_smoothing": False,
    "segmodelparams": {
        # 'mdl_stored_file': '~/lisa_data/liver_intensity.Model.p',
        # "fv_type": 'fv_extern',
        # 'fv_extern':'intensity_localization_fv',
        # 'method': 'multiscale_graphcut',
        "params": {"covariance_type": "full", "n_components": 3}
    },
    "type": "gmmsame",
    "segparams": {"pairwise_alpha_per_mm2":45, "return_only_object_with_seeds": True},
    "series_number": None,
    "slab": {},
    "smoothing_mm": 4,
    "texture_analysis": None,
    "working_voxelsize_mm": 2.0,
    "viewermax": 225,
    "viewermin": -125,
    'segmentation_alternative_params': default_segmentation_alternative_params.copy()

}


