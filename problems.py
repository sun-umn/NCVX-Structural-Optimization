# lint as python3
# Copyright 2019 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# TODO: Need to make a modification to some structures
"""A suite of topology optimization problems."""
# third party
import dataclasses
from typing import Any, Dict, Optional, Union

import numpy as np
import skimage
import torch

from utils import DEFAULT_DEVICE, DEFAULT_DTYPE

X, Y = 0, 1


@dataclasses.dataclass
class Problem:
    """
    Description of a topology optimization problem.
    Attributes:
    normals: float64 array of shape (width+1, height+1, 2) where a value of 1
      indicates a "fixed" coordinate, and 0 indicates no normal force.
    forces: float64 array of shape (width+1, height+1, 2) indicating external
      applied forces in the x and y directions.
    density: fraction of the design region that should be non-zero.
    mask: scalar or float64 array of shape (height, width) that is multiplied by
      the design mask before and after applying the blurring filters. Values of
      1 indicate regions where the material can be optimized; values of 0 are
      constrained to be empty.
    name: optional name of this problem.
    width: integer width of the domain.
    height: integer height of the domain.
    mirror_left: should the design be mirrored to the left when displayed?
    mirror_right: should the design be mirrored to the right when displayed?
    """

    normals: torch.Tensor  # noqa
    forces: torch.Tensor  # noqa
    density: float  # noqa
    x_symmetry: bool = False  # noqa
    y_symmetry: bool = False  # noqa
    mask: Union[torch.Tensor, float] = 1  # noqa
    tounn_mask: Union[None, Dict[str, int]] = None
    name: Optional[str] = None  # noqa
    width: int = dataclasses.field(init=False)  # noqa
    height: int = dataclasses.field(init=False)  # noqa
    mirror_left: bool = dataclasses.field(init=False)  # noqa
    mirror_right: bool = dataclasses.field(init=False)  # noqa

    def __post_init__(self):  # noqa
        self.width = self.normals.shape[0] - 1
        self.height = self.normals.shape[1] - 1

        if self.normals.shape != (self.width + 1, self.height + 1, 2):
            raise ValueError(f"normals has wrong shape: {self.normals.shape}")
        if self.forces.shape != (self.width + 1, self.height + 1, 2):
            raise ValueError(f"forces has wrong shape: {self.forces.shape}")
        if isinstance(self.mask, torch.Tensor) and self.mask.shape != (
            self.height,
            self.width,
        ):
            raise ValueError(f"mask has wrong shape: {self.mask.shape}")

        self.mirror_left = (
            self.normals[0, :, X].all() and not self.normals[0, :, Y].all()
        )
        self.mirror_right = (
            self.normals[-1, :, X].all() and not self.normals[-1, :, Y].all()
        )


# Single-material Structures
def mbb_beam(
    width: int = 60,
    height: int = 20,
    density: float = 0.5,
    device: torch.device = DEFAULT_DEVICE,
    dtype: torch.dtype = DEFAULT_DTYPE,
):
    """Textbook beam example."""
    normals = torch.zeros((width + 1, height + 1, 2)).to(device=device, dtype=dtype)
    normals[-1, -1, Y] = 1
    normals[0, :, X] = 1

    forces = torch.zeros((width + 1, height + 1, 2)).to(device=device, dtype=dtype)
    forces[0, 0, Y] = -1

    return Problem(normals, forces, density)


def mbb_beam_with_circular_non_design_region(
    width=60,
    height=20,
    density=0.5,
    epsilon=1e-3,
    device=DEFAULT_DEVICE,
    dtype=DEFAULT_DTYPE,
):
    """Textbook beam example."""
    normals = torch.zeros((width + 1, height + 1, 2)).to(device=device, dtype=dtype)
    normals[-1, -1, Y] = 1
    normals[0, :, X] = 1

    forces = torch.zeros((width + 1, height + 1, 2)).to(device=device, dtype=dtype)
    forces[0, 0, Y] = -1

    x_center = width // 4
    y_center = height // 2
    radius = y_center // 2

    # Initialize the grid of zeros
    mask = np.ones((height, width))

    # Iterate over each point in the grid
    for y in range(height):
        for x in range(width):
            # Calculate the distance from the center
            distance = np.sqrt((x - x_center) ** 2 + (y - y_center) ** 2)

            # If the distance is within the radius, set the value to 1
            if distance <= radius:
                mask[y, x] = 0

    mask = torch.tensor(mask)
    mask = mask.to(device=device, dtype=dtype)

    return Problem(normals, forces, density, mask)


def cantilever_beam_full(
    width=60,
    height=60,
    density=0.5,
    force_position=0,
    epsilon=1e-3,
    device=DEFAULT_DEVICE,
    dtype=DEFAULT_DTYPE,
):
    """Cantilever supported everywhere on the left"""
    normals = torch.zeros((width + 1, height + 1, 2)).to(device=device, dtype=dtype)
    normals[0, :, :] = 1.0

    forces = torch.zeros((width + 1, height + 1, 2)).to(device=device, dtype=dtype)
    forces[-1, round((1 - force_position) * height), Y] = -1

    return Problem(normals, forces, density)


def cantilever_beam_two_point(
    width=60,
    height=60,
    density=0.5,
    support_position=0.25,
    force_position=0.5,
    device=DEFAULT_DEVICE,
    dtype=DEFAULT_DTYPE,
):
    """Cantilever supported by two points"""
    normals = torch.zeros((width + 1, height + 1, 2)).to(device=device, dtype=dtype)
    normals[0, round(height * (1 - support_position)), :] = 1
    normals[0, round(height * support_position), :] = 1

    forces = torch.zeros((width + 1, height + 1, 2)).to(device=device, dtype=dtype)
    forces[-1, round((1 - force_position) * height), Y] = -1

    # Symmetry requirements
    y_symmetry = True

    return Problem(normals, forces, density, y_symmetry=y_symmetry)


def pure_bending_moment(
    width=60,
    height=60,
    density=0.5,
    support_position=0.45,
    device=DEFAULT_DEVICE,
    dtype=DEFAULT_DTYPE,
):
    """Pure bending forces on a beam."""
    # Figure 28 from
    # http://naca.central.cranfield.ac.uk/reports/arc/rm/3303.pdf
    normals = torch.zeros((width + 1, height + 1, 2)).to(device=device, dtype=dtype)
    normals[-1, :, X] = 1

    # for numerical stability, fix y forces here at 0
    normals[0, round(height * (1 - support_position)), Y] = 1
    normals[0, round(height * support_position), Y] = 1

    forces = torch.zeros((width + 1, height + 1, 2)).to(device=device, dtype=dtype)
    forces[0, round(height * (1 - support_position)), X] = 1
    forces[0, round(height * support_position), X] = -1

    return Problem(normals, forces, density)


def michell_centered_both(
    width=32,
    height=32,
    density=0.5,
    position=0.05,
    device=DEFAULT_DEVICE,
    dtype=DEFAULT_DTYPE,
):
    """A single force down at the center, with support from the side."""
    # https://en.wikipedia.org/wiki/Michell_structures#Examples
    normals = torch.zeros((width + 1, height + 1, 2)).to(device=device, dtype=dtype)
    normals[round(position * width), round(height / 2), Y] = 1
    normals[-1, :, X] = 1

    forces = torch.zeros((width + 1, height + 1, 2)).to(device=device, dtype=dtype)
    forces[-1, round(height / 2), Y] = -1

    return Problem(normals, forces, density)


def michell_centered_below(
    width=32,
    height=32,
    density=0.5,
    position=0.25,
    device=DEFAULT_DEVICE,
    dtype=DEFAULT_DTYPE,
):
    """A single force down at the center, with support from the side below."""
    # https://en.wikipedia.org/wiki/Michell_structures#Examples
    normals = torch.zeros((width + 1, height + 1, 2)).to(device=device, dtype=dtype)
    normals[round(position * width), 0, Y] = 1
    normals[-1, :, X] = 1

    forces = torch.zeros((width + 1, height + 1, 2)).to(device=device, dtype=dtype)
    forces[-1, 0, Y] = -1

    return Problem(normals, forces, density)


def michell_centered_top(
    width=32,
    height=32,
    density=0.5,
    device=DEFAULT_DEVICE,
    dtype=DEFAULT_DTYPE,
):
    """A single force down at the center, with support from the side below."""
    # https://en.wikipedia.org/wiki/Michell_structures#Examples
    normals = torch.zeros((width + 1, height + 1, 2)).to(device=device, dtype=dtype)
    normals[-1, -1, X] = 1
    normals[-1, -1, Y] = 1
    normals[0, -1, X] = 1
    normals[0, -1, Y] = 1

    forces = torch.zeros((width + 1, height + 1, 2)).to(device=device, dtype=dtype)
    forces[round(width // 2), 0, Y] = -1

    return Problem(normals, forces, density)


def ground_structure(
    width=32,
    height=32,
    density=0.5,
    force_position=0.5,
    device=DEFAULT_DEVICE,
    dtype=DEFAULT_DTYPE,
):
    """An overhanging bridge like structure holding up two weights."""
    # https://link.springer.com/content/pdf/10.1007%2Fs00158-010-0557-z.pdf
    normals = torch.zeros((width + 1, height + 1, 2)).to(device=device, dtype=dtype)
    normals[-1, :, X] = 1
    normals[0, -1, :] = 1

    forces = torch.zeros((width + 1, height + 1, 2)).to(device=device, dtype=dtype)
    forces[round(force_position * height), -1, Y] = -1

    return Problem(normals, forces, density)


def l_shape(
    width=32,
    height=32,
    density=0.5,
    aspect=0.4,
    force_position=0.5,
    device=DEFAULT_DEVICE,
    dtype=DEFAULT_DTYPE,
):
    """An L-shaped structure, with a limited design region."""
    # Topology Optimization Benchmarks in 2D
    normals = torch.zeros((width + 1, height + 1, 2)).to(device=device, dtype=dtype)
    normals[: round(aspect * width), 0, :] = 1

    forces = torch.zeros((width + 1, height + 1, 2)).to(device=device, dtype=dtype)
    forces[-1, round((1 - aspect * force_position) * height), Y] = -1

    mask = torch.ones((width, height)).to(device=device, dtype=dtype)
    mask[round(height * aspect) :, : round(width * (1 - aspect))] = 0  # noqa

    # Mask needs to be transposed
    mask = mask.t()

    # For TOuNN
    min_x = round(height * aspect)
    max_x = height
    min_y = 0
    max_y = round(width * (1 - aspect))

    tounn_mask = {'x>': min_x, 'x<': max_x, 'y>': min_y, 'y<': max_y}

    return Problem(normals, forces, density, mask=mask, tounn_mask=tounn_mask)


def crane(
    width=32,
    height=32,
    density=0.3,
    aspect=0.5,
    force_position=0.9,
    device=DEFAULT_DEVICE,
    dtype=DEFAULT_DTYPE,
):
    """A crane supporting a downward force, anchored on the left."""
    normals = torch.zeros((width + 1, height + 1, 2)).to(device=device, dtype=dtype)
    normals[:, -1, :] = 1

    forces = torch.zeros((width + 1, height + 1, 2)).to(device=device, dtype=dtype)
    forces[round(force_position * width), round(1 - aspect * height), Y] = -1

    mask = torch.ones((width, height)).to(device=device, dtype=dtype)
    # the extra +2 ensures that entire region in the vicinity of the force can be
    # be designed; otherwise we get outrageously high values for the compliance.
    mask[round(aspect * width) :, round(height * aspect) + 2 :] = 0  # noqa
    mask = mask.t()

    return Problem(normals, forces, density)


def tower(
    width=32,
    height=32,
    density=0.5,
    device=DEFAULT_DEVICE,
    dtype=DEFAULT_DTYPE,
):
    """A rather boring structure supporting a single point from the ground."""
    normals = torch.zeros((width + 1, height + 1, 2)).to(device=device, dtype=dtype)
    normals[:, -1, Y] = 1
    normals[0, :, X] = 1

    forces = torch.zeros((width + 1, height + 1, 2)).to(device=device, dtype=dtype)
    forces[0, 0, Y] = -1

    return Problem(normals, forces, density)


def center_support(
    width=32,
    height=32,
    density=0.3,
    device=DEFAULT_DEVICE,
    dtype=DEFAULT_DTYPE,
):
    """Support downward forces from the top from the single point."""
    normals = torch.zeros((width + 1, height + 1, 2)).to(device=device, dtype=dtype)
    normals[-1, -1, Y] = 1
    normals[-1, :, X] = 1

    forces = torch.zeros((width + 1, height + 1, 2)).to(device=device, dtype=dtype)
    forces[:, 0, Y] = -1 / width

    return Problem(normals, forces, density)


def column(
    width=32,
    height=32,
    density=0.3,
    device=DEFAULT_DEVICE,
    dtype=DEFAULT_DTYPE,
):
    """Support downward forces from the top across a finite width."""
    normals = torch.zeros((width + 1, height + 1, 2)).to(device=device, dtype=dtype)
    normals[:, -1, Y] = 1
    normals[-1, :, X] = 1

    forces = torch.zeros((width + 1, height + 1, 2)).to(device=device, dtype=dtype)
    forces[:, 0, Y] = -1 / width

    return Problem(normals, forces, density)


def roof(
    width=32,
    height=32,
    density=0.5,
    device=DEFAULT_DEVICE,
    dtype=DEFAULT_DTYPE,
):
    """Support downward forces from the top with a repeating structure."""
    normals = torch.zeros((width + 1, height + 1, 2)).to(device=device, dtype=dtype)
    normals[0, :, X] = 1
    normals[-1, :, X] = 1
    normals[:, -1, Y] = 1

    forces = torch.zeros((width + 1, height + 1, 2)).to(device=device, dtype=dtype)
    forces[:, 0, Y] = -1 / width

    return Problem(normals, forces, density)


def causeway_bridge(
    width=60,
    height=20,
    density=0.3,
    deck_level=1,
    device=DEFAULT_DEVICE,
    dtype=DEFAULT_DTYPE,
):
    """A bridge supported by columns at a regular interval."""
    normals = torch.zeros((width + 1, height + 1, 2)).to(device=device, dtype=dtype)
    normals[-1, -1, Y] = 1
    normals[-1, :, X] = 1
    normals[0, :, X] = 1

    forces = torch.zeros((width + 1, height + 1, 2)).to(device=device, dtype=dtype)
    forces[:, round(height * (1 - deck_level)), Y] = -1 / width

    return Problem(normals, forces, density)


def two_level_bridge(
    width=32,
    height=32,
    density=0.3,
    deck_height=0.2,
    device=DEFAULT_DEVICE,
    dtype=DEFAULT_DTYPE,
):
    """A causeway bridge with two decks."""
    normals = torch.zeros((width + 1, width + 1, 2)).to(device=device, dtype=dtype)
    normals[0, -1, :] = 1
    normals[0, :, X] = 1
    normals[-1, :, X] = 1

    forces = torch.zeros((width + 1, width + 1, 2)).to(device=device, dtype=dtype)
    forces[:, round(height * (1 - deck_height) / 2), :] = -1 / (2 * width)
    forces[:, round(height * (1 + deck_height) / 2), :] = -1 / (2 * width)

    return Problem(normals, forces, density)


def suspended_bridge(
    width=60,
    height=20,
    density=0.3,
    span_position=0.2,
    anchored=False,
    device=DEFAULT_DEVICE,
    dtype=DEFAULT_DTYPE,
):
    """A bridge above the ground, with supports at lower corners."""
    normals = torch.zeros((width + 1, height + 1, 2)).to(device=device, dtype=dtype)
    normals[-1, :, X] = 1
    normals[: round(span_position * width), -1, Y] = 1
    if anchored:
        normals[: round(span_position * width), -1, X] = 1

    forces = torch.zeros((width + 1, height + 1, 2)).to(device=device, dtype=dtype)
    forces[:, -1, Y] = -1 / width

    return Problem(normals, forces, density)


def canyon_bridge(
    width=60,
    height=20,
    density=0.3,
    deck_level=1,
    device=DEFAULT_DEVICE,
    dtype=DEFAULT_DTYPE,
):
    """A bridge embedded in a canyon, without side supports."""
    deck_height = round(height * (1 - deck_level))

    normals = torch.zeros((width + 1, height + 1, 2)).to(device=device, dtype=dtype)
    normals[-1, deck_height:, :] = 1
    normals[0, :, X] = 1

    forces = torch.zeros((width + 1, height + 1, 2)).to(device=device, dtype=dtype)
    forces[:, deck_height, Y] = -1 / width

    return Problem(normals, forces, density)


def multistory_building(
    width=32,
    height=32,
    density=0.3,
    interval=16,
    device=DEFAULT_DEVICE,
    dtype=DEFAULT_DTYPE,
):
    """A multi-story building, supported from the ground."""
    normals = torch.zeros((width + 1, height + 1, 2)).to(device=device, dtype=dtype)
    normals[:, -1, Y] = 1
    normals[-1, :, X] = 1

    forces = torch.zeros((width + 1, height + 1, 2)).to(device=device, dtype=dtype)
    forces[:, ::interval, Y] = -1 / width

    return Problem(normals, forces, density)


def thin_support_bridge(
    width=32,
    height=32,
    density=0.25,
    design_width=0.25,
    device=DEFAULT_DEVICE,
    dtype=DEFAULT_DTYPE,
):
    """
    A bridge supported from below with fixed width supports.
    """
    normals = torch.zeros((width + 1, height + 1, 2)).to(device=device, dtype=dtype)
    normals[:, -1, Y] = 1
    normals[0, :, X] = 1
    normals[-1, :, X] = 1

    forces = torch.zeros((width + 1, height + 1, 2)).to(device=device, dtype=dtype)
    forces[:, 0, Y] = -1 / width

    mask = torch.ones((width, height)).to(device=device, dtype=dtype)
    mask[
        -round(width * (1 - design_width)) :,
        : round(height * (1 - design_width)),  # noqa
    ] = 0  # noqa

    return Problem(normals, forces, density, mask)


def drawbridge(
    width=32,
    height=32,
    density=0.25,
    device=DEFAULT_DEVICE,
    dtype=DEFAULT_DTYPE,
):
    """A bridge supported from above on the left."""
    normals = torch.zeros((width + 1, height + 1, 2)).to(device=device, dtype=dtype)
    normals[0, :, :] = 1

    forces = torch.zeros((width + 1, height + 1, 2)).to(device=device, dtype=dtype)
    forces[:, -1, Y] = -1 / width

    return Problem(normals, forces, density)


def hoop(
    width=32,
    height=32,
    density=0.25,
    device=DEFAULT_DEVICE,
    dtype=DEFAULT_DTYPE,
):
    """Downward forces in a circle, supported from the ground."""
    if 2 * width != height:
        raise ValueError("hoop must be circular")

    normals = torch.zeros((width + 1, height + 1, 2)).to(device=device, dtype=dtype)
    normals[-1, :, X] = 1
    normals[:, -1, Y] = 1

    forces = torch.zeros((width + 1, height + 1, 2)).to(device=device, dtype=dtype)
    i, j, value = skimage.draw.circle_perimeter_aa(
        width, width, width, forces.shape[:2]
    )
    value = torch.tensor(value).to(device=device, dtype=dtype)
    forces[i, j, Y] = -value / (2 * np.pi * width)

    return Problem(normals, forces, density)


def multipoint_circle(
    width=140,
    height=140,
    density=0.333,
    radius=6 / 7,
    weights=(1, 0, 0, 0, 0, 0),
    num_points=12,
    device=DEFAULT_DEVICE,
    dtype=DEFAULT_DTYPE,
):
    """Various load scenarios at regular points in a circle points."""
    # From: http://www2.mae.ufl.edu/mdo/Papers/5219.pdf
    # Note: currently unused in our test suite only because the optimization
    # problems from the paper are defined based on optimizing for compliance
    # averaged over multiple force scenarios.
    c_x = width // 2
    c_y = height // 2
    normals = torch.zeros((width + 1, height + 1, 2)).to(device=device, dtype=dtype)
    normals[c_x - 1 : c_x + 2, c_y - 1 : c_y + 2, :] = 1  # noqa
    assert normals.sum() == 18

    c1, c2, c3, c4, c_x0, c_y0 = weights

    forces = torch.zeros((width + 1, height + 1, 2)).to(device=device, dtype=dtype)
    for position in range(num_points):
        x = radius * c_x * np.sin(2 * np.pi * position / num_points)
        y = radius * c_y * np.cos(2 * np.pi * position / num_points)
        i = int(round(c_x + x))
        j = int(round(c_y + y))
        forces[i, j, X] = +c1 * y + c2 * x + c3 * y + c4 * x + c_x0
        forces[i, j, Y] = -c1 * x + c2 * y + c3 * x - c4 * y + c_y0

    return Problem(normals, forces, density)


def dam(
    width=32,
    height=32,
    density=0.5,
    device=DEFAULT_DEVICE,
    dtype=DEFAULT_DTYPE,
):
    """Support horizitonal forces, proportional to depth."""
    normals = torch.zeros((width + 1, height + 1, 2)).to(device=device, dtype=dtype)
    normals[:, -1, X] = 1
    normals[:, -1, Y] = 1

    forces = torch.zeros((width + 1, height + 1, 2)).to(device=device, dtype=dtype)
    forces[0, :, X] = 2 * torch.arange(1, height + 2) / height**2
    return Problem(normals, forces, density)


def ramp(
    width=32,
    height=32,
    density=0.25,
    device=DEFAULT_DEVICE,
    dtype=DEFAULT_DTYPE,
):
    """Support downward forces on a ramp."""

    return staircase(width=width, height=height, density=density, num_stories=1)


def staircase(
    width=32,
    height=32,
    density=0.25,
    num_stories=2,
    device=DEFAULT_DEVICE,
    dtype=DEFAULT_DTYPE,
):
    """A ramp that zig-zags upward, supported from the ground."""
    normals = torch.zeros((width + 1, height + 1, 2)).to(device=device, dtype=dtype)
    normals[:, -1, :] = 1

    forces = torch.zeros((width + 1, height + 1, 2)).to(device=device, dtype=dtype)
    for story in range(num_stories):
        parity = story % 2
        start_coordinates = (0, (story + parity) * height // num_stories)
        stop_coordiates = (width, (story + 1 - parity) * height // num_stories)
        i, j, value = skimage.draw.line_aa(*start_coordinates, *stop_coordiates)

        value = torch.tensor(value).to(device=device, dtype=dtype)
        forces[i, j, Y] = torch.minimum(forces[i, j, Y], -value / (width * num_stories))

    return Problem(normals, forces, density)


def staggered_points(
    width=32,
    height=32,
    density=0.3,
    interval=16,
    break_symmetry=False,
    device=DEFAULT_DEVICE,
    dtype=DEFAULT_DTYPE,
):
    """A staggered grid of points with downward forces, supported from below."""
    normals = torch.zeros((width + 1, height + 1, 2)).to(device=device, dtype=dtype)
    normals[:, -1, Y] = 1
    normals[0, :, X] = 1
    normals[-1, :, X] = 1

    forces = torch.zeros((width + 1, height + 1, 2)).to(device=device, dtype=dtype)
    f = interval**2 / (width * height)
    # intentionally break horizontal symmetry?
    forces[interval // 2 + int(break_symmetry) :: interval, ::interval, Y] = -f  # noqa
    forces[int(break_symmetry) :: interval, interval // 2 :: interval, Y] = -f  # noqa

    return Problem(normals, forces, density)


# Multi-material Structures
def multi_material_cantilever_beam(
    width: int = 64,
    height: int = 32,
    density: float = 0.6,
    e_materials: torch.Tensor = torch.tensor([3.0, 2.0, 1.0], dtype=torch.double),
    material_density_weight: torch.Tensor = torch.tensor(
        [1.0, 0.7, 0.4], dtype=torch.double
    ),
    device=DEFAULT_DEVICE,
    dtype=DEFAULT_DTYPE,
) -> Dict[str, Any]:
    """
    Problem that defined configuration for multi-material
    cantilever beam
    """
    ndof = 2 * (width + 1) * (height + 1)

    # Forces on the system
    forces = torch.zeros((ndof, 1))
    forces[2 * (width + 1) * (height + 1) - 2 * height + 1, 0] = -1

    # Degrees of freedom
    alldofs_array = np.arange(ndof)

    # Fixed dofs
    fixdofs_array = alldofs_array[0 : 2 * (height + 1) : 1]

    # Free dofs
    freedofs_array = np.sort(list(set(alldofs_array) - set(fixdofs_array)))

    # Convert to torch tensorse)
    mask = torch.tensor(torch.tensor(1.0)).to(device=device, dtype=dtype)
    freedofs = torch.tensor(freedofs_array).to(device=device, dtype=torch.long)
    fixdofs = torch.tensor(fixdofs_array).to(device=device, dtype=torch.long)

    params = {
        # material properties
        "young": 1.0,
        "young_min": 1e-9,
        "poisson": 0.3,
        "g": 0.0,
        # constraints
        "combined_frac": density,
        "volfrac": 0.6,
        "xmin": 0.001,
        "xmax": 1.0,
        # input parameters
        "nelx": torch.tensor(width),
        "nely": torch.tensor(height),
        "mask": mask,
        "freedofs": freedofs,
        "fixdofs": fixdofs,
        "forces": forces.flatten(),
        "penal": 3.0,
        "filter_width": 2,
        "ndof": len(alldofs_array),
        "e_materials": e_materials,
        "material_density_weight": material_density_weight,
        "y_symmetry": False,
        "x_symmetry": False,
        "epsilon": 1e-3,
    }

    return params


def multi_material_bridge(
    width: int = 128,
    height: int = 64,
    density: float = 0.4,
    e_materials: torch.Tensor = torch.tensor([0.2, 0.6, 1.0], dtype=torch.double),
    material_density_weight: torch.Tensor = torch.tensor(
        [0.4, 0.7, 1.0], dtype=torch.double
    ),
    device=DEFAULT_DEVICE,
    dtype=DEFAULT_DTYPE,
) -> Dict[str, Any]:
    """
    Problem that defined configuration for multi-material
    cantilever beam
    """
    ndof = 2 * (width + 1) * (height + 1)

    # Forces on the system
    forces = torch.zeros((ndof, 1))

    forces[2 * (height + 1) * int(width // 4 + 1) - 1, 0] = -1
    forces[2 * (height + 1) * int(2 * width // 4 + 1) - 1, 0] = -2
    forces[2 * (height + 1) * int(3 * width // 4 + 1) - 1, 0] = -1

    # Degrees of freedom
    alldofs_array = np.arange(ndof)

    # Fixed dofs
    fixdofs_array = np.union1d(
        np.array([2 * (height + 1) - 1 - 1, 2 * (height + 1) - 1]),
        np.array([2 * (width + 1) * (height + 1) - 1]),
    )

    # Free dofs
    freedofs_array = np.sort(list(set(alldofs_array) - set(fixdofs_array)))

    # Convert to torch tensorse)
    mask = torch.tensor(torch.tensor(1.0)).to(device=device, dtype=dtype)
    freedofs = torch.tensor(freedofs_array).to(device=device, dtype=torch.long)
    fixdofs = torch.tensor(fixdofs_array).to(device=device, dtype=torch.long)

    params = {
        # material properties
        "young": 1.0,
        "young_min": 1e-9,
        "poisson": 0.3,
        "g": 0.0,
        # constraints
        "combined_frac": density,
        "volfrac": 0.7,
        "xmin": 0.001,
        "xmax": 1.0,
        # input parameters
        "nelx": torch.tensor(width),
        "nely": torch.tensor(height),
        "mask": mask,
        "freedofs": freedofs,
        "fixdofs": fixdofs,
        "forces": forces.flatten(),
        "penal": 3.0,
        "filter_width": 2,
        "ndof": len(alldofs_array),
        "e_materials": e_materials,
        "material_density_weight": material_density_weight,
        "y_symmetry": False,
        "x_symmetry": True,
        "epsilon": 1e-3,
    }

    return params


def build_problems_by_name(device=DEFAULT_DEVICE):
    # Problems Category
    PROBLEMS_BY_CATEGORY = {
        # idealized beam and cantilevers
        "mbb_beam": [
            mbb_beam(96, 32, density=0.5, device=device),
            mbb_beam(192, 64, density=0.5, device=device),
            mbb_beam(384, 128, density=0.3, device=device),
            mbb_beam(384, 128, density=0.5, device=device),
            mbb_beam(192, 32, density=0.5, device=device),
            mbb_beam(384, 64, density=0.4, device=device),
        ],
        "mbb_beam_circular_ndr": [
            mbb_beam_with_circular_non_design_region(
                96, 32, density=0.5, device=device
            ),
            mbb_beam_with_circular_non_design_region(
                192, 64, density=0.4, device=device
            ),
            mbb_beam_with_circular_non_design_region(
                384, 128, density=0.3, device=device
            ),
            mbb_beam_with_circular_non_design_region(
                192, 32, density=0.5, device=device
            ),
            mbb_beam_with_circular_non_design_region(
                384, 64, density=0.4, device=device
            ),
        ],
        "cantilever_beam_full": [
            cantilever_beam_full(96, 32, density=0.4, device=device),
            cantilever_beam_full(192, 64, density=0.3, device=device),
            cantilever_beam_full(352, 96, density=0.4, device=device),
            cantilever_beam_full(384, 128, density=0.4, device=device),
            cantilever_beam_full(384, 128, density=0.2, device=device),
            cantilever_beam_full(384, 128, density=0.15, device=device),
        ],
        "cantilever_beam_two_point": [
            cantilever_beam_two_point(64, 48, density=0.4, device=device),
            cantilever_beam_two_point(96, 96, density=0.4, device=device),
            cantilever_beam_two_point(128, 96, density=0.3, device=device),
            cantilever_beam_two_point(256, 192, density=0.2, device=device),
            cantilever_beam_two_point(256, 192, density=0.15, device=device),
        ],
        "pure_bending_moment": [
            pure_bending_moment(32, 64, density=0.15, device=device),
            pure_bending_moment(64, 128, density=0.125, device=device),
            pure_bending_moment(128, 256, density=0.1, device=device),
        ],
        "michell_centered_both": [
            michell_centered_both(32, 64, density=0.12, device=device),
            michell_centered_both(64, 128, density=0.12, device=device),
            michell_centered_both(128, 256, density=0.12, device=device),
            michell_centered_both(128, 256, density=0.06, device=device),
        ],
        "michell_centered_top": [
            michell_centered_top(32, 64, density=0.12, device=device),
            michell_centered_top(64, 128, density=0.12, device=device),
            michell_centered_top(128, 256, density=0.12, device=device),
            michell_centered_top(128, 256, density=0.06, device=device),
        ],
        "michell_centered_below": [
            michell_centered_below(64, 64, density=0.12, device=device),
            michell_centered_below(128, 128, density=0.12, device=device),
            michell_centered_below(256, 256, density=0.12, device=device),
            michell_centered_below(256, 256, density=0.06, device=device),
        ],
        "ground_structure": [
            ground_structure(64, 64, density=0.12, device=device),
            ground_structure(128, 128, density=0.1, device=device),
            ground_structure(256, 256, density=0.07, device=device),
            ground_structure(256, 256, density=0.05, device=device),
        ],
        # simple constrained designs
        "l_shape_0.2": [
            l_shape(64, 64, aspect=0.2, density=0.4, device=device),
            l_shape(128, 128, aspect=0.2, density=0.3, device=device),
            l_shape(256, 256, aspect=0.2, density=0.2, device=device),
        ],
        "l_shape_0.4": [
            l_shape(64, 64, aspect=0.4, density=0.4, device=device),
            l_shape(128, 128, aspect=0.4, density=0.3, device=device),
            l_shape(192, 192, aspect=0.4, density=0.3, device=device),
            l_shape(256, 256, aspect=0.4, density=0.2, device=device),
        ],
        "crane": [
            crane(64, 64, density=0.3, device=device),
            crane(128, 128, density=0.2, device=device),
            crane(256, 256, density=0.15, device=device),
            crane(256, 256, density=0.1, device=device),
        ],
        # vertical support structures
        "center_support": [
            center_support(64, 64, density=0.15, device=device),
            center_support(128, 128, density=0.1, device=device),
            center_support(256, 256, density=0.1, device=device),
            center_support(256, 256, density=0.05, device=device),
        ],
        "column": [
            column(32, 128, density=0.3, device=device),
            column(64, 256, density=0.3, device=device),
            column(128, 512, density=0.1, device=device),
            column(128, 512, density=0.3, device=device),
            column(128, 512, density=0.5, device=device),
        ],
        "roof": [
            roof(64, 64, density=0.2, device=device),
            roof(128, 128, density=0.15, device=device),
            roof(256, 256, density=0.4, device=device),
            roof(256, 256, density=0.2, device=device),
            roof(256, 256, density=0.1, device=device),
        ],
        # bridges
        "causeway_bridge_top": [
            causeway_bridge(64, 64, density=0.3, device=device),
            causeway_bridge(128, 128, density=0.2, device=device),
            causeway_bridge(256, 256, density=0.1, device=device),
            causeway_bridge(128, 64, density=0.3, device=device),
            causeway_bridge(256, 128, density=0.2, device=device),
        ],
        "causeway_bridge_middle": [
            causeway_bridge(64, 64, density=0.12, deck_level=0.5, device=device),
            causeway_bridge(128, 128, density=0.1, deck_level=0.5, device=device),
            causeway_bridge(256, 256, density=0.08, deck_level=0.5, device=device),
        ],
        "causeway_bridge_low": [
            causeway_bridge(64, 64, density=0.12, deck_level=0.3, device=device),
            causeway_bridge(128, 128, density=0.1, deck_level=0.3, device=device),
            causeway_bridge(256, 256, density=0.08, deck_level=0.3, device=device),
        ],
        "two_level_bridge": [
            two_level_bridge(64, 64, density=0.2, device=device),
            two_level_bridge(128, 128, density=0.16, device=device),
            two_level_bridge(256, 256, density=0.12, device=device),
        ],
        "free_suspended_bridge": [
            suspended_bridge(64, 64, density=0.15, anchored=False, device=device),
            suspended_bridge(128, 128, density=0.1, anchored=False, device=device),
            suspended_bridge(256, 256, density=0.075, anchored=False, device=device),
            suspended_bridge(256, 256, density=0.05, anchored=False, device=device),
        ],
        "anchored_suspended_bridge": [
            suspended_bridge(
                64, 64, density=0.15, span_position=0.1, anchored=True, device=device
            ),
            suspended_bridge(
                128, 128, density=0.1, span_position=0.1, anchored=True, device=device
            ),
            suspended_bridge(
                192,
                192,
                density=0.1,
                span_position=0.1,
                anchored=True,
                device=device,
            ),
            suspended_bridge(
                256,
                256,
                density=0.075,
                span_position=0.1,
                anchored=True,
                device=device,  # noqa
            ),
            suspended_bridge(
                256,
                256,
                density=0.05,
                span_position=0.1,
                anchored=True,
                device=device,  # noqa
            ),
        ],
        "canyon_bridge": [
            canyon_bridge(64, 64, density=0.16, device=device),
            canyon_bridge(128, 128, density=0.12, device=device),
            canyon_bridge(256, 256, density=0.1, device=device),
            canyon_bridge(256, 256, density=0.05, device=device),
        ],
        "thin_support_bridge": [
            thin_support_bridge(64, 64, density=0.3, device=device),
            thin_support_bridge(128, 128, density=0.2, device=device),
            thin_support_bridge(256, 256, density=0.15, device=device),
            thin_support_bridge(256, 256, density=0.1, device=device),
        ],
        "drawbridge": [
            drawbridge(64, 64, density=0.2, device=device),
            drawbridge(128, 128, density=0.15, device=device),
            drawbridge(256, 256, density=0.1, device=device),
        ],
        # more complex design problems
        "hoop": [
            hoop(32, 64, density=0.25, device=device),
            hoop(64, 128, density=0.2, device=device),
            hoop(128, 256, density=0.15, device=device),
        ],
        "dam": [
            dam(64, 64, density=0.2, device=device),
            dam(128, 128, density=0.15, device=device),
            dam(256, 256, density=0.05, device=device),
            dam(256, 256, density=0.1, device=device),
            dam(256, 256, density=0.2, device=device),
        ],
        "ramp": [
            ramp(64, 64, density=0.3, device=device),
            ramp(128, 128, density=0.2, device=device),
            ramp(256, 256, density=0.2, device=device),
            ramp(256, 256, density=0.1, device=device),
        ],
        "staircase": [
            staircase(64, 64, density=0.3, num_stories=3, device=device),
            staircase(128, 128, density=0.2, num_stories=3, device=device),
            staircase(256, 256, density=0.15, num_stories=3, device=device),
            staircase(128, 512, density=0.15, num_stories=6, device=device),
        ],
        "staggered_points": [
            staggered_points(64, 64, density=0.3, device=device),
            staggered_points(128, 128, density=0.3, device=device),
            staggered_points(256, 256, density=0.3, device=device),
            staggered_points(256, 256, density=0.5, device=device),
            staggered_points(64, 128, density=0.3, device=device),
            staggered_points(128, 256, density=0.3, device=device),
            staggered_points(32, 128, density=0.3, device=device),
            staggered_points(64, 256, density=0.3, device=device),
            staggered_points(128, 512, density=0.3, device=device),
            staggered_points(128, 512, interval=32, density=0.15, device=device),
        ],
        "multistory_building": [
            multistory_building(32, 64, density=0.5, device=device),
            multistory_building(64, 128, interval=32, density=0.4, device=device),
            multistory_building(128, 256, interval=64, density=0.3, device=device),
            multistory_building(128, 512, interval=64, density=0.25, device=device),
            multistory_building(128, 512, interval=128, density=0.2, device=device),
        ],
    }

    # Create the ability to do problems by name the same as the paper
    PROBLEMS_BY_NAME = {}
    for problem_class, problem_list in PROBLEMS_BY_CATEGORY.items():
        for problem in problem_list:
            name = f"{problem_class}_{problem.width}x{problem.height}_{problem.density}"
            problem.name = name
            assert name not in PROBLEMS_BY_NAME, f"redundant name {name}"
            PROBLEMS_BY_NAME[name] = problem

    return PROBLEMS_BY_NAME
