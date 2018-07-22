"""
Helper library containing functions that perform spatial geometric operation
like determining if a point is inside a polygon

.. versionadded:: 0.1
.. codeauthor:: Nekhelesh Ramananthan <krnekhelesh@skylarkdrones.com>

**External Dependencies**

.. hlist::
   :columns: 4

   - geopy
   - numpy
   - scipy
   - shapely
"""

# Copyright (C) 2017-2018 Skylark Drones

import math
from collections import OrderedDict
from typing import List, Tuple

import geopy.point
import numpy as np
from geopy.distance import distance, VincentyDistance
from scipy.spatial.qhull import Delaunay
from shapely.geometry import (
    MultiPoint,
    MultiLineString,
    shape,
    LineString,
    Point,
    Polygon,
)
from shapely.ops import cascaded_union, polygonize


def is_point_inside_polygon(point, polygon):
    """
    .. codeauthor:: Nekhelesh Ramananthan <krnekhelesh@gmail.com>

    Determines if a point is *inside* a given polygon or not. A polygon is
    defined as a list of (x, y) tuples.

    :param tuple point: Coordinate of point (x, y)
    :param list(tuple) polygon: Set of points that form a polygon
        [(x1, y1), (x2, y2) ...]
    :return: If point is inside or outside the polygon
    :rtype: bool
    """
    polygon = Polygon(polygon)
    point = Point(point)

    return point.within(polygon)


def _add_edge(edges, edge_points, coords, i, j):
    """
    Add a line between the i-th and j-th points, if not in the list already

    .. codeauthor:: Vaibhav Srinivasa <vaibhav@skylarkdrones.com>
    """
    if (i, j) in edges or (j, i) in edges:
        # already added
        return
    edges.add((i, j))
    edge_points.append(coords[[i, j]])


def exterior_polygon_from_points(points, alpha=1000, buffer=0.0003):
    """
    .. codeauthor:: Vaibhav Srinivasa <vaibhav@skylarkdrones.com>

    Compute the exterior polygon (concave hull) of a set of points.

    .. note::
        The algorithm and workflow was derived from
        http://blog.thehumangeo.com/2014/05/12/drawing-boundaries-in-python/
        
    .. note::
        alpha = 1 / average distance between points in Y direction
    
    Example Input: ::
    
        [
            {'Lat': 16.1235371256305, 'Lng': 75.27075953678545},
            {'Lat': 16.2334347125635, 'Lng': 75.22123563678545}
        ]

    :param list[dict] points: List of coordinate points.
    :param int alpha: alpha value to influence the gooeyness of the border.
        Smaller numbers don't fall inward as much as larger numbers. Too large,
        and you lose everything!
    :param float buffer: Amount of buffer to add/remove from the generated
        polygon. Defaults to 0.0003 (30 meters)
    :return: outer boundary of the concave hull of a set of points
        [(x1, y1), (x2, y2) ...] and area
    :rtype: tuple(list, float)
    """
    data_points = []
    for index, point in enumerate(points):
        data = {
            'properties': OrderedDict([('id', None)]),
            'id': str(index),
            'geometry': {
                'type': 'Point',
                'coordinates': (point['Lng'], point['Lat']),
            },
            'type': 'Feature',
        }
        data_points.append(data)

    shape_points = [shape(points['geometry']) for points in data_points]

    if len(shape_points) < 4:
        # When you have a triangle, there is no sense in computing an alpha
        # shape.
        return MultiPoint(list(shape_points)).convex_hull

    coords = np.array([point.coords[0] for point in shape_points])

    tri = Delaunay(coords)
    edges = set()
    edge_points = []

    # loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.vertices:
        pa = coords[ia]
        pb = coords[ib]
        pc = coords[ic]

        # Lengths of sides of triangle
        a = math.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = math.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = math.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)

        # Semiperimeter of triangle
        s = (a + b + c) / 2.0

        # Area of triangle by Heron's formula
        area = math.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)

        # Here's the radius filter.
        if circum_r < 1.0 / alpha:
            _add_edge(edges, edge_points, coords, ia, ib)
            _add_edge(edges, edge_points, coords, ib, ic)
            _add_edge(edges, edge_points, coords, ic, ia)

    m = MultiLineString(edge_points)
    triangles = list(polygonize(m))

    return (
        list(cascaded_union(triangles).buffer(buffer).exterior.coords),
        cascaded_union(triangles).area,
    )


def calculate_initial_compass_bearing(start_point, end_point):
    """
    .. codeauthor:: Nihal Mohan <nihal@skylarkdrones.com>

    Calculates the initial compass bearing between two points.

    The formulae used is the following:
        θ = atan2(sin(Δlong).cos(lat2), cos(lat1).sin(lat2)
        − sin(lat1).cos(lat2).cos(Δlong))

    :param tuple start_point: Latitude and longitude for the first point in
        decimal degrees
    :param tuple end_point: Latitude and longitude for the second point in
        decimal degrees

    :return: The bearing in degrees
    :rtype: float
    """
    if any(not isinstance(point, tuple) for point in [start_point, end_point]):
        raise TypeError(
            "start_point and end_point must be a tuple of latitude "
            "and longitude"
        )

    start_lat, start_lng = start_point
    end_lat, end_lng = end_point

    start_lat = math.radians(start_lat)
    end_lat = math.radians(end_lat)

    diff_lng = math.radians(end_lng - start_lng)

    x = math.sin(diff_lng) * math.cos(end_lat)
    y = math.cos(start_lat) * math.sin(end_lat) - (
        math.sin(start_lat) * math.cos(end_lat) * math.cos(diff_lng)
    )

    initial_bearing = math.atan2(x, y)

    # Now we have the initial bearing but math.atan2 only returns values
    # from -180° to + 180° which is not what we want for a compass bearing.
    # The solution is to normalize the initial bearing using modulo
    initial_bearing = math.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360

    return compass_bearing


def interpolate_gps_positions(start_point, end_point, interpolate_ratio):
    """
    .. codeauthor:: Nihal Mohan <nihal@skylarkdrones.com>

    Function to interpolate between two GPS Coordinates by a ratio in 2D

    Example Input: ::

            {'Lat': 16.1235371256305, 'Lng': 75.27075953678545, 'Alt': 875.142},
            {'Lat': 16.2334347125635, 'Lng': 75.22123563678545, 'Alt': 893.146},
            0.75



    :param dict start_point: Start point coordinates in decimal degrees
    :param dict end_point: End point coordinates in decimal degrees
    :param float interpolate_ratio: Ratio at which the interpolation should
        happen from the start

    :return: Latitude and longitude of the interpolated GPS points
    :rtype: Tuple(float)
    """
    if any(not isinstance(point, dict) for point in [start_point, end_point]):
        raise TypeError(
            "start_point and end_point inputs should be dictionaries."
            " Refer to documentation!"
        )

    try:
        interpolate_ratio = float(interpolate_ratio)
    except ValueError:
        raise TypeError(
            'Interpolate ratio is required to be a floating value.'
            ' Conversion to float failed!'
        )
    else:
        if interpolate_ratio > 1:
            raise ValueError(
                'Interpolate ratio should be Less than 1. '
                'This is Interpolation. Not Extrapolation!'
            )

    distance_travelled = distance(
        (start_point['Lat'], start_point['Lng']),
        (end_point['Lat'], end_point['Lng']),
    ).meters

    dist_required = distance_travelled * interpolate_ratio

    start = geopy.point.Point(start_point['Lat'], start_point['Lng'])
    d = VincentyDistance(meters=dist_required)
    bearing = calculate_initial_compass_bearing(
        (start_point['Lat'], start_point['Lng']),
        (end_point['Lat'], end_point['Lng']),
    )
    interpolated_point = d.destination(point=start, bearing=bearing)

    # Altitude interpolation if the altitudes were present in the argument
    start_alt = start_point.get('Alt', None)
    end_alt = end_point.get('Alt', None)
    if start_alt is not None and end_alt is not None:
        interpolated_alt = start_alt + (end_alt - start_alt) * interpolate_ratio
        return (
            interpolated_point.latitude,
            interpolated_point.longitude,
            interpolated_alt,
        )
    else:
        return interpolated_point.latitude, interpolated_point.longitude


def split_line_by_length(line, max_length):
    """
    .. codeauthor:: Nekhelesh Ramananthan <krnekhelesh@skylarkdrones.com>
    .. versionadded:: Quark-0.2

    Split a line into segments of lengths not more than maximum length

    :param list(tuples) line: Line composed of points
    :param float max_length:  Maximum length of each segment
    :return: Points that are separated by maximum length provided
    :rtype: Shapely.MultiPoint
    :raise ValueError: If total line length is less than the maximum length
        of segment provided
    """
    line_string = LineString(line)

    total_line_length = 0
    points = list(line_string.coords)
    for index, point in enumerate(points):
        if index < len(points) - 1:
            x1, y1, z1 = point
            x2, y2, z2 = points[index + 1]
            total_line_length += distance((x1, y1), (x2, y2)).meters

    if total_line_length < max_length:
        raise ValueError(
            'Total line length cannot be less than '
            'the maximum length of segment provided'
        )

    splits = math.ceil(total_line_length / max_length)

    return MultiPoint(
        [
            line_string.interpolate((i / splits), normalized=True)
            for i in range(0, splits + 1)
        ]
    )


def interpolate_coordinates_by_fixed_length(
    coords: List[Tuple], interval: float
) -> List[Tuple]:
    """
    .. codeauthor:: Shreehari Murali <hari@skylarkdrones.com>
    .. versionadded:: Quark-0.2

    Interpolates coordinates at fixed interval

    :param list(tuples) coords: Line composed of Coordinates
    :param float interval: The interval at which the linestring coordinates
        have to be present
    :return: Coordinates present at the input interval
    :rtype: list(tuples)
    """
    interpolated_list_of_coords = []
    total_length = 0
    if len(coords) == 0:
        raise ValueError('Coordinates List is Empty!')

    for index, coord in enumerate(coords):
        if len(coord) == 3:  # (lat, long, alt)
            if index < len(coords) - 1:
                lat1, long1, alt1 = coord
                lat2, long2, alt2 = coords[index + 1]
                dist = distance((lat1, long1), (lat2, long2)).meters
        else:  # (lat, long)
            if index < len(coords) - 1:
                lat1, long1 = coord
                lat2, long2 = coords[index + 1]
                dist = distance((lat1, long1), (lat2, long2)).meters
        total_length += dist
    if total_length > interval:
        if len(coord) == 3:
            line_string = split_line_by_length(coords, interval)
        else:
            new_list = []
            for coord in coords:
                lat, long = coord
                new_list.append((lat, long, 0))
            line_string = split_line_by_length(new_list, interval)

        for coord in line_string:
            interpolated_list_of_coords.append((coord.x, coord.y))
        return interpolated_list_of_coords
    else:
        return coords


def buffer_line(
    coords: List[Tuple], buffer_distance: float, interval: float, side: str
) -> List[Tuple]:
    """
    .. codeauthor:: Shreehari Murali <hari@skylarkdrones.com>
    .. versionadded:: Quark-0.2

    Creates a linestring parallel to the given linestring at the given offset.

    :param list(tuples) coords: Line composed of Coordinates in (lat, long) format
    :param float buffer_distance: Parallel offset distance from the input linestring
    :param float interval: The interval at which the linestring coordinates
        have to be present
    :param string side: 'right' or 'left' to the given linestring
    :return: Coordinates buffered at the input interval distance
    :rtype: list(tuples)
    """

    buffered_linestring = (
        LineString(coords)
        .parallel_offset(buffer_distance * 0.00001, side)
        .coords[:]
    )

    return interpolate_coordinates_by_fixed_length(
        buffered_linestring, interval
    )


def project_coord_on_line(point_coord: tuple, coords: List[Tuple]) -> Tuple:
    """
    .. codeauthor:: Shreehari Murali <hari@skylarkdrones.com>
    .. versionadded:: Quark-0.2

    Takes in a coordinate and a linestring and gets the nearest coordinates
    present on the linestring from the input coordinate

    :param tuple point_coord: Coordinate to be projected in
        (lat, long, alt) format
    :param list(tuples) coords: Line containing 3d coordinates
    :return: Coordinate in (lat,long,alt) format
    :rtype: tuple
    """
    projected_coord = 0
    for index, coord in enumerate(coords):
        if index == 0:
            projected_coord = coord
        else:
            least_dist_from_point = distance(
                (projected_coord[0], projected_coord[1]),
                (point_coord[0], point_coord[1]),
            ).meters
            dist_from_point = distance(
                (coord[0], coord[1]), (point_coord[0], point_coord[1])
            ).meters
            if dist_from_point < least_dist_from_point:
                projected_coord = coord

    return projected_coord


def get_point_at_given_length(length: float, coords: List[Tuple]) -> Tuple:
    """
    .. codeauthor:: Shreehari Murali <hari@skylarkdrones.com>
    .. versionadded:: Quark-0.2

    Takes in the linestring containing coordinates and gets the coordinate
    present at the input length

    :param float length: The length of the line at which the coordinate has to
        be obtained.
    :param list(tuples) coords: Line containing coordinates in (lat, long, alt)
        format.
    :return: Coordinate present at the given length in (lat, long, alt) format.
    :rtype: tuple
    """
    current_length = 0
    total_length_of_line = 0

    if length < 0:
        raise ValueError("Invalid Length input!")

    for index, coord in enumerate(coords):
        if index >= len(coords) - 1:
            break
        lat1, long1, alt1 = coord
        lat2, long2, alt2 = coords[index + 1]
        total_length_of_line += distance((lat1, long1), (lat2, long2)).meters

    if length > total_length_of_line:
        raise ValueError(
            "Input Length is larger than " "the length of the line!"
        )
    else:
        for index, coord in enumerate(coords):
            if current_length > length:
                break

            lat1, long1, alt1 = coord
            lat2, long2, alt2 = coords[index + 1]
            current_length += distance((lat1, long1), (lat2, long2)).meters

    return lat1, long1, alt1


def get_points_inside_polygon(
    polygon_coords: List[Tuple], interval: float
) -> List[Tuple]:
    """
    .. codeauthor:: Shreehari Murali <hari@skylarkdrones.com>
    .. versionadded:: Quark-0.2

    Takes in polygon composed of it's exterior coordinates and gets all the
    points inside the polygon at the input interval

    :param list(tuples) polygon_coords: polygon coords present in (lat, long)
        format.
    :param float interval: The interval at which the points have to be present
    :return: Coordinates in (lat, long) format inside the given polygon.
    :rtype: List(tuples)
    """
    # bounding box coordinates
    east = max(polygon_coords, key=lambda coords: coords[1])
    west = min(polygon_coords, key=lambda coords: coords[1])
    north = max(polygon_coords, key=lambda coords: coords[0])
    south = min(polygon_coords, key=lambda coords: coords[0])

    east_linestring = [(south[0], west[1], 0), (north[0], west[1], 0)]
    interpol_east_linestring = interpolate_coordinates_by_fixed_length(
        east_linestring, interval
    )

    dist = 0
    lines = []
    bounding_box_width = distance((east[1], 0), (west[1], 0)).meters

    while dist <= bounding_box_width:
        lines.append(
            buffer_line(interpol_east_linestring, dist, interval, 'left')
        )
        dist += interval
        if dist > bounding_box_width:
            lines.append(
                buffer_line(
                    interpol_east_linestring,
                    bounding_box_width,
                    interval,
                    'left',
                )
            )

    points_inside_bounding_box = []
    for line in lines:
        for coord in line:
            points_inside_bounding_box.append(coord)

    points_inside_polygon = []
    for point in points_inside_bounding_box:
        if is_point_inside_polygon(point, polygon_coords):
            points_inside_polygon.append(point)

    return points_inside_polygon
