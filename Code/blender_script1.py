import bpy # type: ignore
import mathutils # type: ignore
import os
import math
import sys
import enum
import random
PI = math.pi

ASSETS_DIR = os.path.normpath(os.path.join(os.path.dirname(bpy.data.filepath), "../P3Data", "Assets"))

class AssetType(enum.Enum):
    # Asset types: (file_path, obj_name, default_rotation, default_scaling)
    Sedan = ("Vehicles/SedanAndHatchback.blend", "Car", (0, 0, 0), .1)
    StopSign = ("StopSign.blend", "StopSign_Geo", (math.pi/2, 0, math.pi), 1.0)
    TrafficCone = ("TrafficConeAndCylinder.blend", "absperrhut", (math.pi/2, 0, 0), 1.0)
    Pedestrian = ("Pedestrian.blend", "BaseMesh_Man_Simple", (0, 0, 0), 1.0)
    Dustbin = ("Dustbin.blend", "Bin_Mesh.072", (0, 0, 0), 1.0)
    FireHyrant = ("TrafficAssets.blend", "fire", (0, 0, 0), 1.0)
    SmallPole = ("TrafficAssets.blend", "iron pole", (0, 0, 0), 1.0) # Probably for a chain gate or something, probably won't use   
    SpeedLimitSign = ("SpeedLimitSign.blend", "sign_25mph_sign_25mph", (0, 0, 0), 1.0)
    TrafficLight = ("TrafficSignal.blend", "Traffic_signal1", (0, 0, 0), 1.0)
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
        appended_obj = bpy.context.selected_objects[0]  # Get the last object added to the scene

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
        x = random.uniform(-50, 50)
        y = random.uniform(-50, 50)
        car = Asset(car_asset)
        car.place(mathutils.Vector((x, y, 0)))


def main():
    
    clear_scene()

    # Create and place an instance of a sedan
    # sedan = Asset(AssetType.Sedan)
    # sedan.place(mathutils.Vector((20, 0, 0)), additional_rotation=(0, 0, 0))
    
    # Create and place a stop sign
    stop_sign = Asset(AssetType.StopSign)
    stop_sign.place(mathutils.Vector((-2, -2, 0)), additional_rotation=(0,0,0))

    create_random_cars(10)
    
    save_scene(os.path.join(ASSETS_DIR, "..", "script_test.blend"))

if __name__ == "__main__":
    main()