#!/usr/bin/env python

from typing import Any, Iterable
import numpy as np
import tempfile
import warnings
from enum import auto
from io import BytesIO
import math
import PIL
from PIL import Image
from dataclasses import InitVar, dataclass
import click
import geojson
import enum
import itertools
import requests
import urllib.parse
import aiohttp
import asyncio
from osgeo_utils import gdal_merge
from osgeo import gdal
import colorlog

DEBUG = True


handler = colorlog.StreamHandler()
handler.setFormatter(
    colorlog.ColoredFormatter("%(log_color)s%(levelname)s:%(message)s")
)

if DEBUG:
    logger = colorlog.getLogger("console")
    logger.setLevel(colorlog.INFO)
else:
    logger = colorlog.getLogger("debug")
    logger.setLevel(colorlog.DEBUG)

logger.addHandler(handler)


# ------------
# Tiles
# ------------

MAX_TILE_WIDTH_IN_PIXELS = 2048
MAX_TILE_HEIGHT_IN_PIXELS = 2048


@dataclass
class Tile:
    bbox_mu: list[float]  # xmin, ymin, xmax, ymax
    bbox_px: list[float]  # xmin, ymin, xmax, ymax

    _flipped_yaxis: bool = True

    def __post_init__(self):
        assert len(self.bbox_mu) == 4
        assert len(self.bbox_px) == 4

    def to_string(self, bbox: str = "mu") -> str:
        b = self.bbox_mu if bbox == "mu" else self.bbox_px
        return ",".join(str(_) for _ in b)

    def image_dims(self) -> tuple[float, float]:
        return self.bbox_px[2:]

    def pixel_size(self) -> tuple[float]:
        w, h = self.image_dims()
        return w / (self.bbox_mu[2] - self.bbox_mu[0]), h / (
            self.bbox_mu[3] - self.bbox_mu[1]
        )

    def geotransform(self) -> np.array:
        flip = -1 if self._flipped_yaxis else 1
        # 3 non-colinear points are enough to compute the transform
        points_img = [
            [self.bbox_px[0], self.bbox_px[1]],  # Upper-left corner
            [self.bbox_px[2], flip * self.bbox_px[3]],  # Lower-right corner
            [self.bbox_px[0], flip * self.bbox_px[3]],  # Lower-left corner
        ]
        vars = np.pad(points_img, ((0, 0), (1, 0)), constant_values=1)
        A = np.vstack(
            [np.pad(vars, ((0, 0), (0, 3))), np.pad(vars, ((0, 0), (3, 0)))]
        )
        B = [
            self.bbox_mu[0],
            self.bbox_mu[2],
            self.bbox_mu[0],
            self.bbox_mu[1],
            self.bbox_mu[3],
            self.bbox_mu[3],
        ]

        return np.linalg.solve(A, B)


def compute_tilemosaic(bbox: Iterable[float], resolution: int) -> list[Tile]:
    xmin, ymin, xmax, ymax = bbox
    desired_width_px = math.ceil((xmax - xmin) * resolution)
    desired_height_px = math.ceil((ymax - ymin) * resolution)

    # If the requested resolution causes the maximum dimensions
    # defined by X and Y to be exceeded, the region is divided into a
    # set of tiles of the largest possible size.
    if desired_width_px > MAX_TILE_WIDTH_IN_PIXELS:
        tiles_count = desired_width_px // MAX_TILE_WIDTH_IN_PIXELS
        widths_px = [MAX_TILE_WIDTH_IN_PIXELS] * tiles_count

        # Remaining
        if sum(widths_px) < desired_width_px:
            widths_px += [desired_width_px % MAX_TILE_WIDTH_IN_PIXELS]
    else:
        widths_px = [desired_width_px]

    if desired_height_px > MAX_TILE_HEIGHT_IN_PIXELS:
        tiles_count = desired_height_px // MAX_TILE_HEIGHT_IN_PIXELS
        heights_px = [MAX_TILE_HEIGHT_IN_PIXELS] * tiles_count

        # Remaining
        if sum(heights_px) < desired_height_px:
            heights_px += [desired_height_px % MAX_TILE_HEIGHT_IN_PIXELS]
    else:
        heights_px = [desired_height_px]

    widths_mu = [w / resolution for w in widths_px]
    heights_mu = [h / resolution for h in heights_px]

    tiles_dims = list(
        zip(
            # width and heights of each tiles, in pixels
            [(w, h) for h in heights_px for w in widths_px],
            # Cumulative width and heights of each tiles, in map units
            [
                (wmu, hmu)
                for hmu in itertools.accumulate(heights_mu)
                for wmu in itertools.accumulate(widths_mu)
            ],
            # Width and heights of each tiles, in pixels, in map u
            [(wmu, hmu) for hmu in heights_mu for wmu in widths_mu],
        )
    )
    tiles = [
        Tile(
            [
                xmin + dims[1][0] - dims[2][0],  # MinX, in map units
                ymin + dims[1][1] - dims[2][1],  # MinY, in map units
                xmin + dims[1][0],  # MaxX, in map units
                ymin + dims[1][1],  # MaxY, in map units
            ],
            [
                0.0,  # MinX, in pixels
                0.0,  # MinY, in pixels
                dims[0][0],  # MaxX, in pixels
                dims[0][1],  # MaxY, in pixels
            ],
        )
        for dims in tiles_dims
    ]
    return tiles


# ------------
# WMS handler
# ------------


DEFAULT_WMS_OPTIONS = {"SERVICE": "WMS", "VERSION": "1.3.0", "styles": ""}
DEFAULT_WMTS_OPTIONS = {"SERVICE": "WMTS", "VERSION": "1.0.0", "styles": ""}


@dataclass
class WMS:
    endpoint: str
    _cached_capabilities: str = None
    base_options = DEFAULT_WMS_OPTIONS.copy()

    is_wmts: InitVar[bool] = False

    def __post_init__(self, is_wmts: bool) -> None:
        # Run a 'dry' GET query to apply Requests URL validators.
        requests.get(self.endpoint)

        self.is_wmts = is_wmts
        if self.is_wmts:
            self.base_options = DEFAULT_WMTS_OPTIONS.copy()

    def getCapabilities(self) -> str:
        if not self._cached_capabilities:
            # Synthax is Python >= 3.9 only
            params = self.base_options | {"REQUEST": "GetCapabilities"}
            response = requests.get(self.endpoint, params=params)
            response.raise_for_status()
            self._cached_capabilities = response.content
        return self._cached_capabilities

    async def getmap_async(self, tile: Tile, **wms_params: Any) -> bytes:
        w, h = tile.image_dims()
        opts = wms_params | {
            "bbox": tile.to_string(),
            "width": w,
            "height": h,
        }
        params = self.base_options | opts | {"REQUEST": "GetMap"}
        async with self._session.get(
            self.endpoint, params=params, raise_for_status=True
        ) as resp:
            assert resp.status == 200
            bytes_content = await resp.content.read(-1)
            return bytes_content

    async def __aenter__(self) -> "WMS":
        self._session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=5)
        )
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._session:
            await self._session.close()
            # Prevent `ResourceWarning: unclosed transport`
            # See https://docs.aiohttp.org/en/stable/client_advanced.html#graceful-shutdown # noqa
            await asyncio.sleep(0.250)


class WMSError(Exception):
    def __init__(self, xmlerr: str):
        self.message = xmlerr
        super().__init__(self.message)


def get_wms_baseurl(wms_url: str) -> str:
    """Given any URL to a WMS, returns its base URL,
    i.e. the URL without any query, parameter or fragment."""
    return urllib.parse.urljoin(wms_url, urllib.parse.urlparse(wms_url).path)


def is_image_blank(pilimg: Image) -> bool:
    """Returns True is pilimg is plain white."""
    return bool(pilimg.getbbox())


async def download_tile(
    wms: WMS,
    tile: Tile,
    output_dir: str,
    tileid: int = 0,
    skip_blank_tiles=False,
    **wms_params: Any,
) -> str:
    data = await wms.getmap_async(tile, **wms_params)
    try:
        tileimage = Image.open(BytesIO(data))
    except PIL.UnidentifiedImageError as e:
        # The WMS may return a XML error if a WMS parameter was invalid.
        # In such case the data retrieved from getmap() is not a valid image
        # but a byte-encoded XML error.
        raise WMSError(data.decode()) from e

    format = wms_params.get("format").rpartition("/")[-1] or "jpeg"

    should_skip = skip_blank_tiles and is_image_blank(tileimage)
    if not should_skip:
        wf_path = "{}/{}.{}".format(
            output_dir, tileid, get_worldfile_ext(format)
        )
        im_path = "{}/{}.{}".format(output_dir, tileid, format)

        with open(wf_path, "w") as wf:
            wf_data = get_worldfile(tile)
            wf.write(wf_data)

        tileimage.save(
            im_path,
            format=format,
            resampling=Image.BILINEAR,
            quality=100,
        )
        return im_path


async def download_tilemosaic(
    wms: WMS, tiles: list[Tile], output_dir: str, **wms_params: Any
) -> Iterable[str]:
    # Enables async http calls in the WMS client.
    async with wms as async_wms:
        tasks = set()
        for id, tile in enumerate(tiles):
            tiletask = asyncio.create_task(
                download_tile(
                    async_wms,
                    tile,
                    tileid=id,
                    output_dir=output_dir,
                    format="image/jpeg",
                    **wms_params,
                )
            )
            tasks.add(tiletask)
        result = await asyncio.gather(*tasks)
        if not any(result):
            return None
        else:
            return result


# ------------
# Worldfile helpers
# ------------


def get_worldfile(tile: Tile) -> str:
    """
    @See https://en.wikipedia.org/wiki/World_file
    """
    g = tile.geotransform()
    ul_mapunits = [g[0], g[3] + tile.bbox_mu[3] - tile.bbox_mu[1]]
    parameters = [
        g[1],  # A. x-scale
        g[4],  # D. y-skew
        g[2],  # B. x-swew
        g[5],  # E. y-scale
        # C. x coordinate of the center of the image's upper-left pixel
        # in map units
        ul_mapunits[0] + g[1] / 2,
        # F. y coordinate of the center of the image's upper-left pixel
        # in map units
        ul_mapunits[1] + g[5] / 2,
    ]
    return "\n".join(map(str, parameters))


def get_worldfile_ext(format: str) -> str:
    f = format.rpartition("/")[-1]
    ext = f[0] + f[1] + "w"
    return ext


# ------------
# BBOX processing
# ------------


def parse_bbox(ctx, param, value: str) -> list[float]:
    try:
        if value:
            bbox_ = [float(c) for c in value.split(",")]
            assert len(bbox_) == 4
            return bbox_
    except (AssertionError, AttributeError, ValueError) as e:
        raise click.exceptions.BadParameter(
            f"`{value}` is not a valid BBOX."
            "Expected xmin,ymin,xmax,ymax"
            "to be a list of 4 comma-separated numbers.",
            ctx,
            param,
        ) from e


def process_bbox(bbox: list[float]) -> list[list[float]]:
    """Simply wraps a bbox in a list
    For compatibility with process_sheetfile() and process_grid()."""
    return [bbox]


def bbox_from_geojson_feature(feature: dict) -> list[float]:
    coords = np.array(list(geojson.coords(feature)))
    xs = coords[:, 0]
    ys = coords[:, 1]
    return [xs.min(), ys.min(), xs.max(), ys.max()]


def process_sheetfile(
    sheetfile: str, sheetnumber: int, sheetnumbername: str
) -> list[list[float]]:
    """Reads and filters a GeoJSON sheetfile
    and returns the BBOXes for the requested sheetnumber."""
    sheets = geojson.load(sheetfile)["features"]
    if len(sheets) == 0:
        warnings.warn(f"No sheet found in {sheetfile}.", UserWarning)
    else:
        if not sheets[0]["properties"].get(sheetnumbername):
            raise ValueError(
                f"Property {sheetnumbername} not found in sheet features."
            )

    selected = list(
        filter(
            lambda f: f["properties"][sheetnumbername] == sheetnumber, sheets
        )
    )

    if not selected:
        warnings.warn(
            f"No sheet matching {sheetnumbername} = {sheetnumber}.",
            UserWarning,
        )
    elif len(selected) > 1:
        msg = (
            f"{len(selected)} sheets found with number {sheetnumber}.",
            "Each will be processed as a separate region.",
        )
        warnings.warn(msg, UserWarning)

    return [bbox_from_geojson_feature(f) for f in selected]


def process_grid(grid_file: str) -> list[list[float]]:
    """Reads a GeoJSON grid and returns its cells as bounding boxes"""
    cells = geojson.load(grid_file)
    bboxes = []
    for c in cells["features"]:
        bbox = (
            c["properties"]["left"],
            c["properties"]["bottom"],
            c["properties"]["right"],
            c["properties"]["top"],
        )

        try:
            clean = [float(c) for c in bbox]
            bboxes.append(clean)
        except (TypeError, AssertionError) as e:
            raise ValueError(
                "Expected a grid cell with 4 bbox float coordinates"
                " `left`, `bottom`, `right` and `top`but got %s"
                % c["properties"]
            ) from e
    return bboxes


# ------------
# CLI entrypoint
# ------------


class CLIPPING_METHOD(enum.Enum):
    # Keep strategies in order of precedence.
    # Top takes precedence over the other.
    BBOX = auto()
    SHEETFILE = auto()
    GRID = auto()

    def __str__(self) -> str:
        return self.name


def build_output_file_name(fname: str, *suffixes: Iterable[str]) -> str:
    prefix_str = fname.rpartition(".")[0]
    suffix_str = '_'.join(suffixes)
    ext = "tif"
    return ".".join([prefix_str, suffix_str, ext])


@click.command()
@click.argument("layer")
@click.argument(
    "output", type=click.Path(file_okay=True, dir_okay=True, writable=True)
)
@click.option(
    "--wms",
    help="WMS endpoint URL. For instance `https://wms.openstreetmap.fr/wms`.",
)
@click.option(
    "--grid",
    help="""A GEOJSON grid of regions with bboxes as properties
    `{'left': xmin, 'bottom': ymin, 'right': xmax, 'top': ymax}`.
    NOTE: this argument is mutually exclusive with --bbox and --sheetfile.""",
    type=click.File("rb", lazy=True),
)
@click.option(
    "--bbox",
    help="""A bounding box formatted as `xmin,ymin,xmax,ymax`.
    NOTE: this argument is mutually exclusive with --grid and --sheetfile.""",
    type=str,
    callback=parse_bbox,
)
@click.option(
    "--sheetfile",
    help="""A GEOJSON file containing bounding boxes geometries.
    See also --sheetnumber and --sheetnumbername.
    NOTE: this argument is mutually exclusive with --grid and --bbox.""",
    type=click.File("rb", lazy=True),
)
@click.option(
    "--sheetnumber",
    "-n",
    type=click.INT,
    help="Filter a sheetfile to extract only the sheet with this number.",
)
@click.option(
    "--sheetnumbername",
    "-a",
    help="The property on which --sheetfile will filter.",
)
@click.option(
    "--resolution",
    "-r",
    help="""Resolution of the extracted images, in pixels per map unit.
    Defaults to 1.0px/mu. Map units depends on the CRS (see --crs).""",
    default=1.0,
)
@click.option(
    "--crs",
    help="Coordinate reference system, e.g. EPSG:3857.",
    default="EPSG:4326",
)
@click.option(
    "--wmts", help="Use WMTS instead of WMS.", is_flag=True, default=False
)
@click.option(
    "--verbose", "-v", help="Verbose mode", is_flag=True, default=False
)
@click.option("--quiet", "-q", help="Quiet mode", is_flag=True, default=False)
def cli_main(
    layer: str,
    output: click.utils.LazyFile,
    wms: str,
    grid: click.utils.LazyFile,
    sheetfile: click.utils.LazyFile,
    bbox: list[float],
    sheetnumbername: str,
    sheetnumber: int,
    resolution: float,
    wmts: bool,
    crs: str,
    verbose: bool,
    quiet: bool,
):
    """
    LAYER: Name of the WM(T)S layer to scrap.

    \b
    OUTPUT: Path to the GeoTiff file to create.
            The file name will be `{output}.{clipping}[_{region}].tif` where:
                - `clipping` is the name of the clipping method, either "grid",
                "bbox" or "sheetfile";
                - `region` is the index of the corresponding region
                    when clipping multiple regions with a grid or a sheetfile.
    """

    if quiet:
        logger.setLevel(colorlog.ERROR)

    if verbose:
        logger.info("GDAL version %s" % gdal.__version__)

    # ------------
    # Sanity checks and pre-processings on input parameters
    # ------------

    # In case the user provided multiple clipping methods at the same time,
    # only one method is retained based on a preference order defined by
    # the enum class CLIPPING_METHOD.
    # The current order of precedence is: 1. bbox, 2. sheetfile, 3. grid
    input_clipping_opts = {
        CLIPPING_METHOD.BBOX: bbox or False,
        CLIPPING_METHOD.SHEETFILE: sheetfile or False,
        CLIPPING_METHOD.GRID: grid or False,
    }

    if any(input_clipping_opts.values()):
        input_clipping_opts = {
            k: v for k, v in input_clipping_opts.items() if v
        }
        input_clipping_opts = sorted(
            input_clipping_opts, key=lambda m: m.value
        )
        selected_clipping = input_clipping_opts[0]
    else:
        msg = "Expected at least one of --sheetfile, --grid or --bbox."
        raise click.BadArgumentUsage(msg)

    if len(input_clipping_opts) > 1:
        msg = (
            "Setting a region to download with {}"
            "at the same time is not possible."
            "Method {} will take precedence.".format(
                " and ".join(map(str, input_clipping_opts)),
                input_clipping_opts[0],
            )
        )
        warnings.warn(msg, UserWarning)

    # The method SHEETFILE requires options --sheetnumber and -sheetnumbername.
    if selected_clipping == CLIPPING_METHOD.SHEETFILE:
        if not (sheetnumber and sheetnumbername):
            msg = (
                "--sheetfile requires --sheetnumber"
                " and --sheetnumbername to be set."
            )
            raise click.BadArgumentUsage(msg)

    # Check that --wms is an actual WM(T)S endpoint.
    # This is done by checking that the server provides WMS Capabilities.
    # A WMS helper object is built to provide a basic API for later queries.
    baseurl = get_wms_baseurl(wms)
    try:
        wms_obj = WMS(baseurl, is_wmts=wmts)
        wms_obj.getCapabilities()
    except requests.exceptions.RequestException as exc:
        raise click.exceptions.BadParameter(
            f"Invalid WMS URL `{baseurl}`"
        ) from exc

    if verbose:
        logger.info(
            "Endpoint: %s (WMTS=%s)" % (wms_obj.endpoint, wms_obj.is_wmts)
        )
        logger.info("Selected clipping method: %s" % selected_clipping.name)
    # ------------
    # Step 1: resolve the tiles to download
    # ------------

    # For each region a set of tiles is prepared.
    # The number of tile depends on:
    #   - the region extent
    #   - the requested resolution
    #   - the constraints on the tiles dimensions
    #      See MAX_TILE_[WIDTH|HEIGHT]_IN_PIXELS.
    if selected_clipping == CLIPPING_METHOD.BBOX:
        # 1 region
        bboxes = process_bbox(bbox)
        region_names = [""]
    elif selected_clipping == CLIPPING_METHOD.SHEETFILE:
        #  1 or several regions
        bboxes = process_sheetfile(sheetfile, sheetnumber, sheetnumbername)
        region_names = [f"{sheetnumber}_{i}" for i in range(len(bboxes))]
    elif selected_clipping == CLIPPING_METHOD.GRID:
        #  1 or several regions
        bboxes = process_grid(grid)
        region_names = list(range(1, len(bboxes) + 1))
    else:
        raise NotImplementedError(
            f"{selected_clipping} is not yet implemented"
        )

    regions = [compute_tilemosaic(b, resolution) for b in bboxes]

    # ------------
    # Step 2: fetch the tiles and merge all tiles in each region
    #         into a single GeoTiff
    # ------------
    loop = asyncio.get_event_loop_policy().get_event_loop()
    for ix, mosaic in enumerate(regions):
        with tempfile.TemporaryDirectory() as odir:
            if verbose:
                logger.info(
                    "Downloading %i tiles in region %s."
                    % (len(mosaic), region_names[ix] or layer)
                )

            r = loop.run_until_complete(
                download_tilemosaic(
                    wms_obj, mosaic, odir, crs=crs, layers=layer
                )
            )

            ofile = build_output_file_name(
                output,
                selected_clipping.__str__().lower(),
                region_names[ix].__str__().lower(),
            )
            if not r:
                warnings.warn(f"No tile to build {ofile}.")
            else:
                parameters = [
                    "",
                    "-o",
                    ofile,
                    "-co",
                    "COMPRESS=LZW",
                ]

                if DEBUG or verbose:
                    parameters.append("-v")

                if verbose:
                    logger.info(
                        "Writing region %i to %s with GDAL parameters `%s`"
                        % (ix, ofile, " ".join(parameters))
                    )

                gdal_merge.main(parameters + r)


def entrypoint() -> None:
    warnings.filterwarnings("error")
    if DEBUG:
        try:
            cli_main()
        except Exception as e:
            logger.exception(e)
            raise e
    else:
        try:
            cli_main()
        except Warning as w:
            logger.warning(w)
        except Exception as e:
            logger.error(e)


if __name__ == "__main__":
    entrypoint()
