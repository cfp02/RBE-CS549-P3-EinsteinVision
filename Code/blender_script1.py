import bpy # type: ignore
import mathutils # type: ignore
import os
import math
import sys
import enum
import random
import time
import json
PI = math.pi

ASSETS_DIR = os.path.normpath(os.path.join(os.path.dirname(bpy.data.filepath), "../P3Data", "Assets"))


class AssetController:
    '''
    In charge of holding all of the Assets in the scene and managing actions related to them
    '''

    def __init__(self, json_file=None):
        self.frames : dict[int, list[Asset]] = {}
        if json_file is not None:
            self.json_to_assets(json_file)
            # Created all assets in memory
            
            #Place the first frame's objects
            self.place_assets(min(self.frames.keys()))


    def add_assets(self, assets, frame=0):
        if frame not in self.frames:
            self.frames[frame] = []
        self.frames[frame].extend(assets)
        # print("Added these assets: ", assets, " to frame: ", frame)

    def place_assets(self, frame=0):
        if frame in self.frames:
            for asset in self.frames[frame]:
                asset.place(asset.location)
    
    def json_to_assets(self, json_file):
        json_reader = JSONReader(json_file)
        print(len(json_reader.data))
        for frame_data in json_reader.data:
            frame = frame_data['frame']
            assets = frame_data['assets']
            asset_list = []
            for asset_json in assets:
                # Create the asset object
                asset = Asset(self.asset_type_from_string(asset_json['type']), 
                              location=mathutils.Vector(asset_json['location']),
                              rotation=mathutils.Euler(asset_json['rotation'], 'XYZ'),
                              scaling=asset_json['scaling'])
                asset_list.append(asset)
            self.add_assets(asset_list, frame)

    def asset_type_from_string(self, type_str):
        asset_type = None
        # Phase 1 includes only lanes, vehicles, pedestrians, traffic lights, stop signs
        match type_str:
            case "Sedan":
                asset_type = AssetType.Sedan
            case "StopSign":
                asset_type = AssetType.StopSign
            case "TrafficCone":
                asset_type = AssetType.TrafficCone
            case "Pedestrian":
                asset_type = AssetType.Pedestrian
            case "Dustbin":
                asset_type = AssetType.Dustbin
            case "FireHyrant":
                asset_type = AssetType.FireHyrant
            case "SmallPole":
                asset_type = AssetType.SmallPole
            case "SpeedLimitSign":
                asset_type = AssetType.SpeedLimitSign
            case "TrafficLight":
                asset_type = AssetType.TrafficLight
            case _:
                print("Asset type not recognized. You're gonna be a fire hydrant! --", type_str)
                asset_type = AssetType.FireHyrant

        return asset_type


class AssetType(enum.Enum):
    # Asset types: (file_path, obj_name, default_rotation, default_scaling)
    Sedan = ("Vehicles/SedanAndHatchback.blend", "Car", (0, 0, 0), .12)
    StopSign = ("StopSign.blend", "StopSign_Geo", (math.pi/2, 0, math.pi/2), 2.0)
    TrafficCone = ("TrafficConeAndCylinder.blend", "absperrhut", (math.pi/2, 0, 0), 10.0)
    Pedestrian = ("Pedestrain.blend", "BaseMesh_Man_Simple", (math.pi/2, 0, 0), .055)
    Dustbin = ("Dustbin.blend", "Bin_Mesh.072", (PI/2, 0, 0), 10)
    FireHyrant = ("TrafficAssets.blend", "Circle.002", (0, 0, 0), 1.5)
    SmallPole = ("TrafficAssets.blend", "Cylinder.001", (0, 0, 0), 1.0) # Probably for a chain gate or something, probably won't use   
    SpeedLimitSign = ("SpeedLimitSign.blend", "sign_25mph_sign_25mph", (0, 0, 0), 4.0)
    TrafficLight = ("TrafficSignal.blend", "Traffic_signal1", (PI/2, 0, 0), 1.5)
    # Lane markings

    def __init__(self, file_path, obj_name, default_rotation, default_scaling):
        self.file_path = file_path
        self.obj_name = obj_name
        self.default_rotation = default_rotation
        self.default_scaling = default_scaling

class Asset:
    def __init__(self, asset_type: AssetType, location=None, rotation=None, scaling=None):
        self.asset_type = asset_type
        self.location = location if location is not None else mathutils.Vector((0, 0, 0))
        print("This is the location: ", self.location)
        self.rotation = rotation if rotation is not None else self.asset_type.default_rotation
        self.scaling = scaling * self.asset_type.default_scaling if scaling is not None else self.asset_type.default_scaling
        self.id = None
        print("Created Asset of type: ", asset_type)

    def place(self, location, assets_dir = ASSETS_DIR, additional_rotation=(0, 0, 0), scaling=None):
        self.location = location
        if scaling is not None:
            self.scaling = self.scaling * scaling
        # Apply default rotation and additional rotation
        total_rotation = [d + a for d, a in zip(self.asset_type.default_rotation, additional_rotation)]
        self.rotation = mathutils.Euler(total_rotation, 'XYZ')
        
        # Load the asset
        file_path = os.path.join(assets_dir, self.asset_type.file_path)
        bpy.ops.wm.append(filename="Object/" + self.asset_type.obj_name, directory=file_path, link=False)
        
        # Position the asset
        # appended_obj = bpy.data.objects.get(self.asset_type.obj_name)
        try:
            appended_obj = bpy.context.selected_objects[0]  # Get the last object added to the scene
        except IndexError:
            print("Error: No object found from object: ", file_path, self.asset_type.obj_name)

        if appended_obj:
            appended_obj.location = self.location
            appended_obj.rotation_euler = self.rotation
            appended_obj.scale = (self.scaling, self.scaling, self.scaling)
            self.id = appended_obj

def save_scene(filepath):
    bpy.ops.wm.save_as_mainfile(filepath=filepath)

def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

def create_random_cars(num_cars, car_asset : AssetType = AssetType.Sedan):
    for i in range(num_cars):
        x = random.uniform(-500, 50)
        y = random.uniform(-50, 50)
        car = Asset(car_asset)
        car.place(mathutils.Vector((x, y, 0)))

def create_all_assets(random_placement=False):
    x_spacing = 5
    for i, asset_type in enumerate(AssetType):
        if random_placement:
            x = random.uniform(-50, 50)
        else:
            x = -50 + (i + 1) * x_spacing  # Calculate x position based on uniform spacing
        y = 5
        asset = Asset(asset_type)
        asset.place(mathutils.Vector((x, y, 0)))

class JSONReader:
    """
    JSON File in the format of:

    [
        {
            "frame": 1,
            "assets": [
                {
                    "type": "Sedan",
                    "location": [0, 0, 0],
                    "rotation": [0, 0, 0],
                    "scaling": 1.0
                },
                {
                    "type": "StopSign",
                    "location": [0, 0, 0],
                    "rotation": [0, 0, 0],
                    "scaling": 1.0
                }
            ]
        }
    ]

    """
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None
        self.read()

    def read(self):
        print("Reading JSON file: ", self.filepath)
        with open(self.filepath, 'r') as file:
            self.data = json.load(file)
        print("Finished reading JSON file, data: ", self.data)




def main():
    
    clear_scene()
    
    # # Create and place a stop sign
    # stop_sign = Asset(AssetType.StopSign)
    # stop_sign.place(mathutils.Vector((-2, -2, 0)), additional_rotation=(0,0,0))

    # create_all_assets()
    # create_random_cars(10)

    print("Creating assetcontroller")
    asset_controller = AssetController('assets.json')

    
        
    save_scene(os.path.join(ASSETS_DIR, "..", "script_test.blend"))
    print("Finished")

if __name__ == "__main__":
    main()