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
    def __init__(self, asset_type: AssetType):
        self.asset_type = asset_type
        self.location = mathutils.Vector((0, 0, 0))
        self.rotation = mathutils.Euler(asset_type.default_rotation, 'XYZ')
        self.scaling = asset_type.default_scaling
        self.id = None

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
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None
        self.read()

    def read(self):
        with open(self.filepath, 'r') as file:
            self.data = json.load(file)

    def get_asset(self, asset_id):
        asset_data = self.data.get(asset_id)
        if asset_data is None:
            return None
        asset_type = AssetType[asset_data["type"]]
        asset = Asset(asset_type)
        asset.location = mathutils.Vector(asset_data["location"])
        asset.rotation = mathutils.Euler(asset_data["rotation"], 'XYZ')
        asset.scaling = asset_data["scaling"]
        return asset

    def get_all_assets(self):
        assets = []
        for asset_id in self.data.keys():
            asset = self.get_asset(asset_id)
            if asset is not None:
                assets.append(asset)
        return assets

    def get_asset_ids(self):
        return list(self.data.keys())

    def get_asset_data(self, asset_id):
        return self.data.get(asset_id)

    def get_all_asset_data(self):
        return self.data

    def get_asset_type(self, asset_id):
        asset_data = self.get_asset_data(asset_id)
        if asset_data is None:
            return None
        return AssetType[asset_data["type"]]

    def get_asset_location(self, asset_id):
        asset_data = self.get_asset_data(asset_id)
        if asset_data is None:
            return None
        return mathutils.Vector(asset_data["location"])

    def get_asset_rotation(self, asset_id):
        asset_data = self.get_asset_data(asset_id)
        if asset_data is None:
            return None
        return mathutils.Euler(asset_data["rotation"], 'XYZ')

    def get_asset_scaling(self, asset_id):
        asset_data = self.get_asset_data(asset_id)
        if asset_data is None:
            return None
        return asset_data["scaling"]

    def get_asset_ids_by_type(self, asset_type):
        asset_ids = []
        for asset_id in self.data.keys():
            if self.get_asset_type(asset_id) == asset_type:
                asset_ids.append(asset_id)
        return asset_ids

    def get_asset_ids_by_location(self, location):
        asset_ids = []
        for asset_id in self.data.keys():
            if self.get_asset_location(asset_id) == location:
                asset_ids.append(asset_id)
        return asset


def main():
    
    clear_scene()
    
    # # Create and place a stop sign
    # stop_sign = Asset(AssetType.StopSign)
    # stop_sign.place(mathutils.Vector((-2, -2, 0)), additional_rotation=(0,0,0))

    create_all_assets()
    # create_random_cars(10)

    
    save_scene(os.path.join(ASSETS_DIR, "..", "script_test.blend"))

if __name__ == "__main__":
    main()