import isx
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.measure import label, regionprops
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from ipywidgets import interactive, widgets, Output, Layout
from IPython.display import clear_output, display
from typing import Any, Optional
import matplotlib.colors as mcolors
import pandas as pd
import concurrent.futures
from typing import Dict
from pathlib import Path
import json

# Constants
ZERO_VALUE = 0

def read_image(path: str) -> np.ndarray:
    """
    Reads an image from a given path.

    Parameters:
        path: The path of the image file.

    Returns:
        The image data.
    """
    return isx.Image.read(path).get_data()


def process_with_n_slider(
    sizing_widget: widgets.VBox,
    img_demean: Any,
    LoG_footprints: Any,
    cell_diameter: int,
):
    """
    Process the footprints with a slider to adjust the enlargement.

    This function creates a slider to adjust the enlargement of the footprints, and processes the footprints
    based on the slider value.

    Parameters:
    sizing_widget (widgets.VBox): The widget containing the sliders for size adjustment.
    img_demean (Any): The demeaned image.
    LoG_footprints (Any): The footprints from the Laplacian of Gaussian image.
    cell_diameter (int): The diameter of the cells.

    Returns:
    The final footprints after processing.
    """

    def process_footprints(
        sizing_widget: widgets.VBox,
        n_value: int,
        img_demean: Any,
        LoG_footprints: Any,
        cell_diameter: float,
    ) -> None:
        """
        Process the footprints based on the slider values and the 'n' value.

        This function calculates the min and max sizes based on the slider values, filters the footprints,
        splits and merges the cells, and displays the final footprints.

        Parameters:
        sizing_widget (widgets.VBox): The widget containing the sliders for size adjustment.
        n_value (int): The 'n' value for merging cells.
        img_LoG (Any): The Laplacian of Gaussian image.
        img_demean (Any): The demeaned image.
        LoG_footprints (Any): The footprints from the Laplacian of Gaussian image.
        cell_diameter (float): The diameter of the cells.

        Returns:
        None
        """

        nonlocal final_footprints, final_footprints_copy

        kernel_size = cell_diameter * 2
        img_LoG = LoG_convolve(img_demean, kernel_size=kernel_size)
        max_size_multiplier = sizing_widget.children[0].children[0].value
        min_size_multiplier = sizing_widget.children[0].children[1].value

        average_cell_area = np.pi * (cell_diameter**2) / 4
        min_size = int(average_cell_area * min_size_multiplier)
        max_size = int(average_cell_area * max_size_multiplier)

        small_footprints = footprint_filter(img_LoG, LoG_footprints, max_size, min_size)

        max_size_multiplier2 = sizing_widget.children[0].children[2].value
        min_size_multiplier2 = sizing_widget.children[0].children[0].value

        min_size = int(average_cell_area * min_size_multiplier2)
        max_size = int(average_cell_area * max_size_multiplier2)

        medium_footprints = footprint_filter(
            img_demean, LoG_footprints, max_size, min_size
        )

        split_footprints, _ = split_cells(
            img_demean, LoG_footprints, medium_footprints, cell_size=cell_diameter * 0.8
        )

        final_footprints_copy = merge_cells(small_footprints, split_footprints, n=0)
        final_footprints = merge_cells(small_footprints, split_footprints, n=n_value)
        with out:
            clear_output(wait=True)
            show_img(
                img_demean,
                final_footprints,
                title="Final footprints",
                plot_individual=True,
            )

    def on_n_change(change: Any) -> None:
        """
        Handle changes in the 'n' slider value.

        This function is triggered when the 'n' slider value changes. It clears the output and re-processes
        the footprints based on the new 'n' value.

        Parameters:
        change (widgets.ValueChangeHandler): The instance containing the new value and other information.

        Returns:
        None
        """
        with out:
            clear_output(wait=True)
            process_footprints(
                sizing_widget,
                change.new,
                img_demean,
                LoG_footprints,
                cell_diameter,
            )

    final_footprints = []
    final_footprints_copy = []
    out = Output()

    n_slider = widgets.IntSlider(
        value=0, min=0, max=10, step=1, description="Enlarge by N pixel:"
    )
    n_slider.observe(on_n_change, names="value")

    display(n_slider)  # Display the slider
    display(out)
    process_footprints(
        sizing_widget,
        n_slider.value,
        img_demean,
        LoG_footprints,
        cell_diameter,
    )
    return n_slider, np.array(final_footprints_copy)


def sizing_with_sliders(
    img_demean: np.ndarray,
    threshold_widget,
    cell_diameter: int,
) -> widgets.VBox:
    """
    Create an interactive plot with sliders to adjust the size of the footprints.

    This function creates sliders to adjust the size multipliers, a button to show the final footprints,
    and an interactive plot that updates the image based on the slider values.

    Parameters:
    img_demean (Any): The demeaned image.
    img_LoG (Any): The Laplacian of Gaussian image.
    LoG_footprints (Any): The footprints from the Laplacian of Gaussian image.
    cell_diameter (int): The diameter of the cells.

    Returns:
    widgets.VBox: A VBox widget containing the interactive plot and the output widget.
    """
    LoG_footprints = []

    def update_images(
        max_size_multiplier: float,
        min_size_multiplier: float,
        max_size_multiplier2: float,
    ) -> None:
        """
        Update the image based on the provided size multipliers.

        This function filters the footprints based on their size, splits multiple-cell ROIs with a watershed filter,
        and merges the footprints from the two categories.

        Parameters:
        max_size_multiplier (float): The multiplier for the maximum size of the footprints.
        min_size_multiplier (float): The multiplier for the minimum size of the footprints.
        max_size_multiplier2 (float): The second multiplier for the maximum size of the footprints.

        Returns:
        None
        """
        nonlocal final_footprints, out, img_shown, button, threshold_widget, img_demean, cell_diameter, LoG_footprints

        # Update the processed image upon slider adjustments

        kernel_size = cell_diameter * 2
        img_LoG = LoG_convolve(img_demean, kernel_size=kernel_size)

        LoG_footprints0 = threshold_img(
            img_LoG, std_factor=threshold_widget.children[0].children[0].value
        )
        LoG_footprints1 = shrink_footprints(
            img_demean,
            LoG_footprints0,
            threshold=threshold_widget.children[0].children[1].value,
        )
        LoG_footprints = footprint_filter(
            img_LoG,
            LoG_footprints1,
            max_size=1000000,
            min_size=int(np.pi * (cell_diameter * cell_diameter) * 0.1),
        )

        with out:
            clear_output(wait=True)

            average_cell_area = np.pi * (cell_diameter**2) / 4
            min_size = int(average_cell_area * min_size_multiplier)
            max_size = int(average_cell_area * max_size_multiplier)

            small_footprints = footprint_filter(
                img_LoG, LoG_footprints, max_size, min_size
            )

            # Adjust the size multiplier to detect overlapping footprints
            min_size = max_size
            max_size = int(average_cell_area * max_size_multiplier2)

            medium_footprints = footprint_filter(
                img_demean, LoG_footprints, max_size, min_size
            )

            # Split multiple-cell ROIs with watershed filter
            split_footprints, _ = split_cells(
                img_demean,
                LoG_footprints,
                medium_footprints,
                cell_size=cell_diameter * 0.8,
            )

            # Disable the sliders during the delay
            max_size_multiplier_slider.disabled = True
            min_size_multiplier_slider.disabled = True
            max_size_multiplier2_slider.disabled = True

            show_img(img_demean, split_footprints, plot_individual=True)
            final_footprints = merge_cells(small_footprints, split_footprints, n=1)

            max_size_multiplier_slider.disabled = False
            min_size_multiplier_slider.disabled = False
            max_size_multiplier2_slider.disabled = False

            img_shown = False
            button.description = "Show final footprints"

    def show_image_on_click(b: Any) -> None:
        """
        Show or hide the image when the button is clicked.

        This function creates a colored mask of the final footprints and overlays it on the image.
        The mask is colored according to the unique values in the third dimension of the indices.

        Parameters:
        b (Any): The button instance that triggered the event.

        Returns:
        None
        """
        nonlocal img_shown
        mask_3d = final_footprints
        indices = np.argwhere(mask_3d == 1)

        unique_values = np.unique(indices[:, 2])
        system_colors = list(
            mcolors._colors_full_map.values()
        )  # Get a list of all colors

        # color_dict = {val: plt.cm.tab20(i) for i, val in enumerate(unique_values)}
        color_dict = {val: system_colors[i] for i, val in enumerate(unique_values)}

        colors = np.array([color_dict[val] for val in indices[:, 2]])

        rects = [Rectangle((idx[1], idx[0]), 1, 1) for idx in indices]
        pc = PatchCollection(rects, edgecolors=colors, alpha=0.6)
        with out:
            if not img_shown:
                fig, ax = plt.subplots()
                ax.imshow(img_demean, cmap="gray", vmax=np.percentile(img_demean, 99))
                ax.add_collection(pc)
                ax.autoscale()
                ax.axis("off")
                plt.show()
                img_shown = True
                button.description = "Hide All Images"
            else:
                clear_output()
                img_shown = False
                button.description = "Show Final Footprints"

    max_size_multiplier_slider = widgets.FloatSlider(
        value=1.2, min=1, max=5, description="Size to split"
    )
    min_size_multiplier_slider = widgets.FloatSlider(
        value=0.4, min=0, max=2, description="Min Size Multiplier"
    )
    max_size_multiplier2_slider = widgets.FloatSlider(
        value=8, min=5, max=10, description="Max Size Multiplier2"
    )

    out = widgets.Output(layout=Layout(margin="0 0 0 30px"))
    final_footprints = []
    img_shown = False
    button = widgets.Button(description="Show Final footprints")

    interactive_plot = interactive(
        update_images,
        max_size_multiplier=max_size_multiplier_slider,
        min_size_multiplier=min_size_multiplier_slider,
        max_size_multiplier2=max_size_multiplier2_slider,
    )

    button.on_click(show_image_on_click)

    display(interactive_plot, out, button)
    return widgets.VBox([interactive_plot, out]), LoG_footprints


def threshold_with_sliders(img_demean: Any, cell_diameter: int) -> widgets.VBox:
    """
    Create interactive sliders to adjust the threshold of an image.

    Parameters:
    img_LoG (Any): The input image after Laplacian of Gaussian (LoG) filtering.
    img_demean (Any): The demeaned input image.
    cell_diameter (int): The estimated cell diameter.

    Returns:
    widgets.VBox: A VBox widget containing the interactive sliders and the output image.
    """

    out = widgets.Output()
    LoG_footprints = []
    n_std = 0.3
    image_threshold = 0.1

    n_std_slider = widgets.FloatSlider(
        value=n_std, min=0, max=3, step=0.05, description="n_std:"
    )
    image_threshold_slider = widgets.FloatSlider(
        value=image_threshold, min=0, max=1, step=0.01, description="image_threshold:"
    )

    def update_image(n_std: float, image_threshold: float) -> None:
        nonlocal LoG_footprints, cell_diameter, img_demean
        kernel_size = cell_diameter * 2
        img_LoG = LoG_convolve(img_demean, kernel_size=kernel_size)
        with out:
            clear_output(wait=True)
            LoG_footprints_temp = threshold_img(img_LoG, std_factor=n_std)
            LoG_footprints_filtered = shrink_footprints(
                img_demean, LoG_footprints_temp, threshold=image_threshold
            )
            LoG_footprints = footprint_filter(
                img_LoG,
                LoG_footprints_filtered,
                max_size=1000000,
                min_size=int(np.pi * (cell_diameter * cell_diameter) * 0.1),
            )
            title = f"LoG filtered ROIs that are brighter than {image_threshold*100}% of the raw image maximal luminance"
            show_img(img_demean, LoG_footprints, title=title)

    threshold_widgets = interactive(
        update_image, n_std=n_std_slider, image_threshold=image_threshold_slider
    )

    return widgets.VBox([threshold_widgets, out])


def convolve_with_slider(
    img_demean: Any, title: str = "Image Title", percentile: int = 99
) -> widgets.VBox:
    """
    Create an interactive slider to convolve an image with a Laplacian of Gaussian (LoG) filter.

    Parameters:
    img_demean (Any): The input image.
    title (str, optional): The title of the image. Defaults to 'Image Title'.
    percentile (int, optional): The percentile to use when displaying the image. Defaults to 99.

    Returns:
    widgets.VBox: A VBox widget containing the interactive slider and the output image.
    """

    cell_diameter = 12
    n = 2  # default 2
    kernel_size = cell_diameter * n
    img_LoG = LoG_convolve(img_demean, kernel_size=kernel_size)

    out = widgets.Output(layout=Layout(margin="0 0 0 30px"))

    cell_diameter_slider = widgets.IntSlider(
        value=cell_diameter, min=1, max=50, step=1, description="Cell Diameter:"
    )

    def on_value_change(cell_diameter: int) -> None:
        nonlocal img_LoG
        nonlocal out
        nonlocal n
        kernel_size = cell_diameter * n
        img_LoG = LoG_convolve(img_demean, kernel_size=kernel_size)
        title = "showing image after LoG"
        with out:
            clear_output(wait=True)
            show_img(img_LoG, title=title, percentile=99)

    interactive_plot = interactive(on_value_change, cell_diameter=cell_diameter_slider)

    return widgets.VBox([interactive_plot, out])


def show_img(
    img_matrix: np.ndarray,
    footprints: np.ndarray = None,
    title: str = None,
    percentile: int = 99,
    plot_individual: bool = False,
    color: str = "red",
) -> None:
    """
    Function shows an image and (optionally) overlays contour/footprint map.

    Parameters:
        img_matrix: A background image.
        footprints: A 2D contour map or 3D cell footprints.
        title: Image title.
        percentile: Used to set colormap vmax value to control the image brightness.
        plot_individual: Plot one footprint at a time, slower compared to contour plot.
        color: Color of the contour lines.

    Returns:
        None
    """

    plt.figure(figsize=(8, 5))
    plt.imshow(
        img_matrix,
        interpolation="nearest",
        cmap="gray",
        vmax=np.percentile(img_matrix, percentile),
    )

    if footprints is not None and np.any(footprints):
        if footprints.ndim > 2 and plot_individual:
            for cell in range(footprints.shape[2]):
                plt.contour(
                    footprints[:, :, cell], colors=color, linewidths=0.2, alpha=0.4
                )
        elif not plot_individual:
            if footprints.ndim > 2:
                footprints = np.max(footprints, axis=2)
            plt.contour(footprints, colors=color, linewidths=0.2, alpha=0.4)
    if title is not None:
        if footprints is not None and footprints.size == 0:
            plt.title("no footprints detected")
        else:
            plt.title(title)
    # if title is not None:
    #     plt.title(
    #         f"no footprints detected"
    #         if footprints is not None or not np.any(footprints)
    #         else title
    # )

    plt.xticks([])
    plt.yticks([])
    plt.show()


def generate_LoG_filter(kernel_size: int) -> np.ndarray:
    """
    Function generates a Laplacian of Gaussian (LoG) filter.
    The filter generation and convolution procedure were implemented
    based on functions described at
    https://projectsflix.com/opencv/laplacian-blob-detector-using-python/

    Parameters:
        kernel_size : int
            Size of the kernel for the LoG filter.

    Returns:
        LoG filter with a given kernel size.
    """

    kernel_size = int(np.ceil(kernel_size))
    sigma = kernel_size / 6
    y, x = np.mgrid[
        -kernel_size // 2 : kernel_size // 2 + 1,
        -kernel_size // 2 : kernel_size // 2 + 1,
    ]
    dist_sq = x**2 + y**2

    LoG_filter = (
        (-(2 * sigma**2) + dist_sq)
        * np.exp(-dist_sq / (2.0 * sigma**2))
        * (1 / (2 * np.pi * sigma**4))
    )

    return LoG_filter


def LoG_convolve(img_matrix: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Function convolves image with a LoG filter with given kernel size
    and normalizes the output image to 16 bit scale.

    Parameters:
        img_matrix: Input image.
        kernel_size: The kernel size for the LoG filter.

    Returns:
        Convolved and normalized image.
    """

    # Generate LoG filter
    filter_LoG = generate_LoG_filter(kernel_size)

    # Convolve image with the filter
    image = cv2.filter2D(img_matrix, -1, filter_LoG)

    # Square the response
    image = np.square(image)

    # Normalize the image to 16 bit scale
    img_LoG_norm = cv2.normalize(image, None, 0, 65535, cv2.NORM_MINMAX)

    return img_LoG_norm


def mask_around_center(
    img_matrix: np.ndarray,
    center: tuple[int, int],
    max_radius: int,
    min_radius: int = 0,
) -> np.ndarray:
    """
    Function draws a circular mask confined by the two distance boundaries from a given center.

    Parameters:
        img_matrix: Input image matching the footprint image size.
        center: Cell centroid x and y coordinates.
        max_radius: Outer boundary.
        min_radius: Inner boundary.

    Returns:
        Circular or donut shaped ROIs confined by the two distance boundaries from the center.
    """

    h, w = img_matrix.shape
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.hypot(X - center[0], Y - center[1])
    mask = np.logical_and(
        dist_from_center <= max_radius, dist_from_center >= min_radius
    )

    return mask


def ring_contrast(
    img_matrix: np.ndarray, footprint: np.ndarray, cell_size: int, n: int = 2
) -> float:
    """
    Function calculates the luminance contrast by measuring the average cell body luminance and
    dividing it by the average ring background luminance.

    Parameters:
        img_matrix: Input image matching the footprint image size.
        footprint: 2D boolean matrix, a single cell footprint.
        cell_size: Cell size defining the cell body.
        n: Background confined by cell_size and cell_size+n.

    Returns:
        The average luminance ratio between the cell body and the background.
    """

    cell_size_extra = cell_size + n

    centroids = ndimage.measurements.center_of_mass(footprint)
    y0, x0 = np.round(centroids).astype(int)

    center_mask = mask_around_center(img_matrix, center=(x0, y0), max_radius=cell_size)
    peri_mask = mask_around_center(
        img_matrix, center=(x0, y0), max_radius=cell_size_extra, min_radius=cell_size
    )

    return (np.mean(img_matrix[center_mask]) / np.mean(img_matrix[peri_mask])).astype(
        float
    )


def footprint_filter(
    img_matrix: np.ndarray,
    footprints: np.ndarray,
    max_size: int,
    min_size: int,
    average_size: int = None,
    contrast: float = 0.5,
) -> np.ndarray:
    """
    Function applies cell size and contrast (optional) constraints to filter footprints.

    Parameters:
        img_matrix: Input image matching the footprint image size.
        footprints: 3D footprints matrix (h x w x cell_index).
        max_size: Maximal size.
        min_size: Minimal size. Cell size filter limiting cell selection to the between range.
        average_size: Optional. Used to calculate body-to-background contrast.
        contrast: Set the minimal body-to-background luminance contrast.

    Returns:
        Footprint subsets after size and contrast filtering.
    """

    if np.any(footprints):
        cell_filter = [
            np.logical_and.reduce(
                [
                    np.sum(footprint) < max_size,
                    np.sum(footprint) > min_size,
                    (
                        ring_contrast(img_matrix, footprint, average_size) > contrast
                        if average_size is not None
                        else True
                    ),
                ]
            )
            for footprint in np.dsplit(footprints, footprints.shape[2])
        ]

        return footprints[:, :, cell_filter]
    else:
        return None


def contour_to_footprint(img_matrix: np.ndarray, contour: np.ndarray) -> np.ndarray:
    """
    Function converts contour to footprints.

    Parameters:
        img_matrix: Input image matching the contour image size.
        contour: Threshold contour.

    Returns:
        Segmented footprints.
    """

    labels = label(contour)
    props = regionprops(labels)
    h, w = img_matrix.shape[0], img_matrix.shape[1]
    footprints = np.zeros((h, w, len(props)))

    for i, prop in enumerate(props, start=1):
        # option1
        xy_coord = prop.coords
        footprints[xy_coord[:, 0], xy_coord[:, 1], i - 1] = True

        # option2
        # footprints[np.where(labels == i)] = True

    return footprints


def threshold_img(img_matrix: np.ndarray, std_factor: float) -> np.ndarray:
    """
    Function applies an intensity threshold to detect suprathreshold footprints.

    Parameters:
        img_matrix: Image to be thresholded.
        std_factor: Multiplier for standard deviation based threshold.

    Returns:
        Suprathreshold footprints.
    """

    # Calculate the threshold
    threshold = np.mean(img_matrix) + np.std(img_matrix) * std_factor

    # Apply the threshold
    _, contour = cv2.threshold(img_matrix, threshold, 65535, cv2.THRESH_BINARY)

    # Convert the contour to footprints
    footprints = contour_to_footprint(img_matrix, contour)

    return footprints


def expand_footprints(
    img_matrix: np.ndarray, footprints: np.ndarray, threshold: float = 0.7
) -> np.ndarray:
    """
    Function adds suprathreshold ROIs from the input image to given footprints.

    Parameters:
        img_matrix: Raw image before LOG convolution.
        footprints: Footprints to be expanded.
        threshold: Raw image threshold.

    Returns:
        Expanded footprints.
    """

    # Calculate the suprathreshold contour
    supertheshold_contour = img_matrix >= np.max(img_matrix) * threshold

    # Calculate the LoG contour
    LoG_contour = np.max(footprints, axis=2)

    # Combine the contours
    expanded_contour = np.logical_or(supertheshold_contour, LoG_contour)

    # Convert the contour to footprints
    expanded_footprints = contour_to_footprint(img_matrix, expanded_contour)

    return expanded_footprints


def shrink_footprints(
    img_matrix: np.ndarray, footprints: np.ndarray, threshold: float = 0.7
) -> Optional[np.ndarray]:
    """
    Function intersects suprathreshold ROIs with given footprints.

    Parameters:
        img_matrix: Raw image before LOG convolution.
        footprints: Input footprints.
        threshold: Raw image threshold.

    Returns:
        Shrank footprints.
    """

    if np.any(footprints):
        # Calculate the suprathreshold contour
        supertheshold_contour = img_matrix >= np.max(img_matrix) * threshold

        # Calculate the LoG contour
        LoG_contour = np.max(footprints, axis=2)

        # Intersect the contours
        shrink_contour = np.logical_and(supertheshold_contour, LoG_contour)

        # Convert the contour to footprints
        shrink_footprints = contour_to_footprint(img_matrix, shrink_contour)

        return shrink_footprints
    else:
        return None


def split_cells(
    img_matrix: np.ndarray,
    LoG_footprints: np.ndarray,
    medium_footprints: np.ndarray,
    cell_size: int,
):
    """
    Function splits medium size footprints based on watershed filter using local maximal Euclidean distance
    and merges with differential footprints sets (LoG_footprints-medium_footprints)

    Parameters:
        img_matrix: input image matching the footprint image size
        LoG_footprints: super sets
        medium_footprints: medium size subsets
        cell_size: threshold to split medium size footprints

    Returns:
        split, and merged footprints
    """

    if np.any(medium_footprints):
        # compute the exact Euclidean distance from every binary
        # pixel to the nearest zero pixel, then find peaks in this
        # distance map
        mask = np.max(medium_footprints, axis=2)
        D = ndimage.distance_transform_edt(mask)
        coords = peak_local_max(
            D,
            min_distance=int(cell_size),
            footprint=np.ones((3, 3)),
            labels=mask.astype(int),
        )
        mask2 = np.zeros(D.shape, dtype=bool)
        mask2[tuple(coords.T)] = True
        markers, _ = ndimage.label(mask2)
        labels = watershed(-D, markers, mask=mask)

        # detect differential footprints that are not medium sized from expanded sets
        medium_size_contour = np.max(medium_footprints, axis=2)
        expanded_contour = np.max(LoG_footprints, axis=2)
        diff_contour = (expanded_contour == 1) & (medium_size_contour == 0)
        diff_footprints = contour_to_footprint(img_matrix, diff_contour)

        # merge the footprints from split sets to difference sets
        split_footprints = contour_to_footprint(img_matrix, labels)
        final_footprints = np.dstack((diff_footprints, split_footprints))
    else:
        final_footprints = LoG_footprints
        split_footprints = np.empty(shape=(0, 0))
    return split_footprints, final_footprints


def enlarge_footprints(
    final_footprints: np.ndarray, n_slider: widgets.IntSlider
) -> np.ndarray:
    """
    Merges two sets of footprints.

    Parameters:
        n: Enlarge the footprints by n pixels using dilation function.

    Returns:
        Enlarged footprints.
    """
    n = n_slider.value
    if final_footprints.size != 0:
        for i in range(final_footprints.shape[2]):
            img = final_footprints[:, :, i]
            kernel = np.ones((2 * n + 1, 2 * n + 1), np.uint8)
            img_dilation = cv2.dilate(img, kernel, iterations=1)
            final_footprints[:, :, i] = img_dilation

    return final_footprints


def merge_cells(
    small_footprints: np.ndarray, split_footprints: np.ndarray, n: int = 2
) -> np.ndarray:
    """
    Merges two sets of footprints.

    Parameters:
        n: Enlarge the footprints by n pixels using dilation function.
        split_footprints: Medium size footprints after splitting.
        small_footprints: Small size footprints.

    Returns:
        Merged footprints.
    """
    if split_footprints.size != 0:
        merged_footprints = np.concatenate((split_footprints, small_footprints), axis=2)
        final_footprints = merged_footprints
    else:
        final_footprints = small_footprints

    if final_footprints.size != 0:
        for i in range(final_footprints.shape[2]):
            img = final_footprints[:, :, i]
            kernel = np.ones((2 * n + 1, 2 * n + 1), np.uint8)
            img_dilation = cv2.dilate(img, kernel, iterations=1)
            final_footprints[:, :, i] = img_dilation

    return final_footprints


def footprints_export_to_isxd(
    image_path, isxd_path, footprints: np.ndarray, suffix: str
) -> str:
    """
    Exports footprints to an ISXD file.

    Parameters:
        input_isxd: Reference image to create cell sets and copy metadata.
        footprints: Sets to be exported.
        suffix: Suffix to automatically name output ISXD file.

    Returns:
        Footprint cellsets in ISXD format.
    """

    
    if isxd_path is not None:
        input_isxd = Path(isxd_path)
    else:
        input_isxd = Path(image_path)
    output_path = Path.cwd() / f"{input_isxd.stem}{suffix}.isxd"
    movie = isx.Movie.read(str(input_isxd))
    if output_path.exists():
        output_path.unlink()

    cell_set = isx.CellSet.write(
        str(output_path),
        timing=isx.Timing(num_samples=1),
        spacing=movie.spacing
    )

    for i in range(footprints.shape[2]):
        image = footprints[:, :, i].astype(np.float32)
        trace = np.empty(1).astype(np.float32)
        cell_set.set_cell_data(i, image, trace, "C{}".format(i))

    cell_set.flush()
    del cell_set

    copy_metadata(str(input_isxd), str(output_path))
    print(f"A new file has been generated: {str(output_path)}")
    return str(output_path)


def read_metadata(
    filename: str, sizeof_size_t: int = 8, endianness: str = "little"
) -> Dict:
    """
    Reads json metadata stored in an .isxd file

    Parameters:
        filename: The .isxd filename
        sizeof_size_t: Number of bytes used to represent a size_t type variable in C++
        endianness: Endianness of your machine

    Returns:
        Metadata represented as a json dictionary
    """

    with open(filename, "rb") as infile:
        # Inspired by isxJsonUtils.cpp
        infile.seek(-sizeof_size_t, 2)
        header_size = int.from_bytes(infile.read(sizeof_size_t), endianness)
        bottom_offset = header_size + 1 + sizeof_size_t
        infile.seek(-bottom_offset, 2)
        string_json = infile.read(bottom_offset - sizeof_size_t - 1).decode("utf-8")
        json_metadata = json.loads(string_json)

    return json_metadata


def write_metadata(
    filename: str,
    json_metadata: Dict,
    sizeof_size_t: int = 8,
    endianness: str = "little",
) -> None:
    """
    Writes json metadata to an .isxd file

    Parameters:
        filename: The .isxd filename
        json_metadata: Metadata represented as a json dictionary
        sizeof_size_t: Number of bytes used to represent a size_t type variable in C++
        endianness: Endianness of your machine
    """

    with open(filename, "rb+") as infile:
        infile.seek(-sizeof_size_t, 2)
        header_size = int.from_bytes(infile.read(sizeof_size_t), endianness)
        bottom_offset = header_size + 1 + sizeof_size_t
        infile.seek(-bottom_offset, 2)

        infile.truncate()
        # process non-ascii characters correctly
        string_json = json.dumps(json_metadata, indent=4, ensure_ascii=False) + "\0"
        infile.write(bytes(string_json, "utf-8"))

        # calculate number of bytes in string by encoding to utf-8
        string_json = string_json.encode("utf-8")
        json_length = int.to_bytes(len(string_json) - 1, sizeof_size_t, endianness)
        infile.write(json_length)


def copy_metadata(input_isxd: str, autoseg_isxd: str) -> None:
    """
    Copies metadata from input_isxd to autoseg_isxd file.

    Parameters:
        input_isxd: Reference file that contains the metadata to be copied.
        autoseg_isxd: Newly generated isxd cellmap file, the metadata copy target.
    """

    # Read metadata for autoseg isxd cell set
    autoseg_metadata = read_metadata(autoseg_isxd)

    # Read metadata of input isxd movie
    movie_metadata = read_metadata(input_isxd)

    # Overwrite extra properties of autoseg isxd cell set
    # The extraProperties key in the metadata is where information about the acquisition settings is stored
    # e.g., miniscope type (dual color, multiplane, etc.)
    autoseg_metadata["extraProperties"] = movie_metadata["extraProperties"]

    # Write the updated metadata back to the autoseg isxd file
    write_metadata(autoseg_isxd, autoseg_metadata)