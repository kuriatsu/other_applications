#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from mcap.reader import make_reader
from rclpy.serialization import deserialize_message

from perception_msgs.msg import AugmentedScene

def draw_selected_map(ax, segment_ids, map, color):
    # extract lane id from path
    draw_id_list = []
    for segment_id in segment_ids:
        lane_data = map.iloc[int(np.where(map.lane_id==str(segment_id))[0][0])]
        for id in [segment_id, lane_data.left_neighbor_lane_id, lane_data.right_neighbor_lane_id]:
            if id != "": draw_id_list.append(id)

    draw_id_list = set(draw_id_list)

    for lane_id in draw_id_list:
        geometry = map.iloc[int(np.where(map.lane_id==str(id))[0][0])].geometry
        gpd.GeoSeries(geometry).plot(ax=ax, color=color, alpha=0.2)

def draw_line(ax, points, color, label, style):
    line_coords = []
    for point in points:
    # Define coordinates for a simple line
        line_coords.append((point[0], point[1]))

    # Create a LineString geometry
    line_geometry = LineString(line_coords)
    point_geometry = []
    for p in points:
        point_geometry.append(Point((p[0], p[1])))
    # Create a GeoDataFrame
    gdf_line = gpd.GeoDataFrame(geometry=[line_geometry, *point_geometry])
    gdf_line.plot(color=color, ax=ax, label=label, linestyle=style, markersize=10)

    # Create start point

def process_planned_trajectoty(msg):
    return {
        "header": {
            "frame_id": msg.header.frame_id,
            "measurement_timestamp": msg.header.measurement_timestamp,
            "creation_timestamp": msg.header.creation_timestamp,
        },
        "trajectory_points": [
            {
                "position": {
                    "x": point.position.x,
                    "y": point.position.y,
                    "z": point.position.z,
                },
                "velocity": point.velocity,
                "acceleration": point.acceleration,
                "relative_time": point.relative_time,
            }
            for point in msg.trajectory_points
        ],
    }

def process_debug_prediction(msg):
    return {
        "header": {
            "frame_id": msg.header.frame_id,
            "measurement_timestamp": msg.header.measurement_timestamp,
            "creation_timestamp": msg.header.creation_timestamp,
        },
        "predicted_trajectories": [
            {
                "object_id": traj.object_id,
                "trajectory_points": [
                    {
                        "position": {
                            "x": point.position.x,
                            "y": point.position.y,
                            "z": point.position.z,
                        },
                        "velocity": point.velocity,
                        "acceleration": point.acceleration,
                        "relative_time": point.relative_time,
                    }
                    for point in traj.trajectory_points
                ],
                "probability": traj.probability,
            }
            for traj in msg.predicted_trajectories
        ],
    }

def process_ego_lane_info(msg):
    return {
        "header": {
            "frame_id": msg.header.frame_id,
            "measurement_timestamp": msg.header.measurement_timestamp,
            "creation_timestamp": msg.header.creation_timestamp,
        },
        "lane_segments": [
            {
                "lane_id": segment.lane_id,
                "left_neighbor_lane_id": segment.left_neighbor_lane_id,
                "right_neighbor_lane_id": segment.right_neighbor_lane_id,
                "priority": segment.priority,
                "length": segment.length,
                "width": segment.width,
                "speed_limit": segment.speed_limit,
                "lane_type": segment.lane_type,
                "turn_type": segment.turn_type,
                "junction_id": segment.junction_id,
                "road_markings": [
                    {
                        "type": marking.type,
                        "color": marking.color,
                        "width": marking.width,
                        "material": marking.material,
                    }
                    for marking in segment.road_markings
                ],
                "geometry_points": [
                    {
                        "x": point.x,
                        "y": point.y,
                        "z": point.z,
                    }
                    for point in segment.geometry_points
                ],
            }
            for segment in msg.lane_segments
        ],
    }

def process_augmented_objects(msg):
    return {
        "header": {
            "frame_id": msg.header.frame_id,
            "measurement_timestamp": msg.header.measurement_timestamp,
            "creation_timestamp": msg.header.creation_timestamp,
        },
        "augmented_objects": [
            {
                "bbox_info": {
                    "id": obj.bbox_info.id,
                    "position": {
                        "x": obj.bbox_info.position.x,
                        "y": obj.bbox_info.position.y,
                        "z": obj.bbox_info.position.z,
                    },
                    "local_position": {
                        "x": obj.bbox_info.local_position.x,
                        "y": obj.bbox_info.local_position.y,
                        "z": obj.bbox_info.local_position.z,
                    },
                    "theta": obj.bbox_info.theta,
                    "local_theta": obj.bbox_info.local_theta,
                    "length": obj.bbox_info.length,
                    "width": obj.bbox_info.width,
                    "height": obj.bbox_info.height,
                    "type": obj.bbox_info.type,
                    "timestamp": obj.bbox_info.timestamp,
                    "confidence": obj.bbox_info.confidence,
                    "sub_type": obj.bbox_info.sub_type,
                    "sensor_type": obj.bbox_info.sensor_type,
                    },
                "tracking_info": {
                    "velocity": {
                        "x": obj.tracking_info.velocity.x,
                        "y": obj.tracking_info.velocity.y,
                        "z": obj.tracking_info.velocity.z,
                    },
                    "local_relative_velocity": {
                        "x": obj.tracking_info.local_relative_velocity.x,
                        "y": obj.tracking_info.local_relative_velocity.y,
                        "z": obj.tracking_info.local_relative_velocity.z,
                    },
                    "local_velocity": {
                        "x": obj.tracking_info.local_velocity.x,
                        "y": obj.tracking_info.local_velocity.y,
                        "z": obj.tracking_info.local_velocity.z,
                    },
                    "polygon_point": [
                        {
                            "x": p.x,
                            "y": p.y,
                            "z": p.z,
                        }
                        for p in obj.tracking_info.polygon_point
                    ],
                    "track_measurement_timestamp": obj.tracking_info.track_measurement_timestamp,
                    "acceleration": {
                        "x": obj.tracking_info.acceleration.x,
                        "y": obj.tracking_info.acceleration.y,
                        "z": obj.tracking_info.acceleration.z,
                    },
                    "local_acceleration": {
                        "x": obj.tracking_info.local_acceleration.x,
                        "y": obj.tracking_info.local_acceleration.y,
                        "z": obj.tracking_info.local_acceleration.z,
                    },
                    "tangental_acceleration": obj.tracking_info.tangental_acceleration,
                },
                "augmentor_info": {
                    "assigned_nth_lanes": [
                        {
                            "index": lane.index,
                            "probability": lane.probability,
                        }
                        for lane in obj.augmentor_info.assigned_nth_lanes
                    ],
                    "assigned_lane_segments": [
                        {
                            "index": seg.index,
                            "probability": seg.probability,
                        }
                        for seg in obj.augmentor_info.assigned_lane_segments
                    ],
                },
            }
            for obj in msg.objects
        ],
    }

def validate(plan_msg, augmented_msg, prediction_msg, lane_map_msg, behavioral_msg):


def main(mcap_file):
    with open(mcap_file, "rb") as f:
        reader = make_reader(f)
        plan_msg = None
        augmented_msg = None
        prediction_msg = None
        lane_map_msg = None
        behavioral_msg = None

        for _schema, channel, message in reader.iter_messages():
            if channel.topic == "/t2/planning/planned_trajectory":
                plan_msg = deserialize_message(message.data, PlannedTrajectory)
                print("PlannedTrajectory:", ros_msg)
            if channel.topic == "/t2/planning/debug_behavioral":
                behavioral_msg = deserialize_message(message.data, DebugBehavioral)
                print("LaneMap:", ros_msg)
            if channel.topic == "/t2/object_augmentor/augmented_scene":
                augmented_msg = deserialize_message(message.data, AugmentedScene)
                print("AugmentedObjects:", ros_msg)
            if channel.topic == "/t2/planning/debug_prediction":
                prediction_msg = deserialize_message(message.data, DebugPrediction)
                print("Prediction:", ros_msg)
            if channel.topic == "/lane_creator/ego_lane_info":
                lane_map_msg = deserialize_message(message.data, EgoLaneInfo)
                print("LaneMap:", ros_msg)

            if plan_msg and augmented_msg and prediction_msg and lane_map_msg and behavioral_msg:
                processed_plan = validate(plan_msg, augmented_msg, prediction_msg, lane_map_msg, behavioral_msg)
                plan_msg = []

if __name__ == "__main__":
    main("/home/kuribayashi-a/data/log/record_develop_7.mcap")
