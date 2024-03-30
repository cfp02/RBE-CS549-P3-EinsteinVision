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
BASE_PATH = os.path.normpath(os.path.join(os.path.dirname(bpy.data.filepath), "../Output"))
JSON_DATA_PATH = os.path.normpath(os.path.join(os.path.dirname(bpy.data.filepath),"../JSONData"))

class AssetController:
    '''
    In charge of holding all of the Assets in the scene and managing actions related to them
    '''

    def __init__(self, json_file=None):
        self.frames : dict[int, list[Asset]] = {}
        json_path = os.path.join(JSON_DATA_PATH, json_file)
        if json_path is not None:
            self.json_to_assets(json_path)
            # Created all assets in memory

    def place_first_frame(self):
        self.place_assets(min(self.frames.keys()))


    def add_assets(self, assets, frame=0):
        if frame not in self.frames:
            self.frames[frame] = []
        self.frames[frame].extend(assets)
        # print("Added these assets: ", assets, " to frame: ", frame)

    def place_assets(self, frame=0):
        if frame in self.frames:
            for asset in self.frames[frame]:
                asset.place(asset.location, additional_rotation=asset.rotation)
    
    def json_to_assets(self, json_file):
        json_reader = JSONReader(json_file)
        print(len(json_reader.data))
        for frame_data in json_reader.data:
            frame = frame_data['frame']
            # assets = frame_data['assets']
            assets = frame_data['objects']
            asset_list = []
            for asset_json in assets:
                # Create the asset object
                asset = Asset(self.asset_type_from_string(asset_json['type']), 
                              location=mathutils.Vector([coord * 50 for coord in asset_json['location']]),
                              rotation=mathutils.Euler(asset_json['rotation'], 'XYZ'),
                              scaling=asset_json['scaling'])
                asset_list.append(asset)
            self.add_assets(asset_list, frame)

    def asset_type_from_string(self, type_str):
        asset_type = None
        # Phase 1 includes only lanes, vehicles, pedestrians, traffic lights, stop signs
        match type_str:
            case "Sedan" | 'car' | 'truck':
                asset_type = AssetType.Sedan
            case "StopSign" | "stop sign":
                asset_type = AssetType.StopSign
            case "TrafficCone":
                asset_type = AssetType.TrafficCone
            case "Pedestrian" | 'person':
                asset_type = AssetType.Pedestrian
            case "Dustbin":
                asset_type = AssetType.Dustbin
            case "FireHyrant":
                asset_type = AssetType.FireHyrant
            case "SmallPole":
                asset_type = AssetType.SmallPole
            case "SpeedLimitSign":
                asset_type = AssetType.SpeedLimitSign
            case "TrafficLight" | "traffic light":
                asset_type = AssetType.TrafficLight
            case _:
                print("Asset type not recognized. You're gonna be a fire hydrant! --", type_str)
                asset_type = AssetType.FireHyrant

        return asset_type


class AssetType(enum.Enum):
    # Asset types: (file_path, obj_name, default_rotation, default_scaling, texture_path=None)
    Sedan = ("Vehicles/SedanAndHatchback.blend", "Car", (0, 0, 0), .12, None)
    StopSign = ("StopSign.blend", "StopSign_Geo", (math.pi/2, 0, math.pi/2), 2.0, os.path.join(ASSETS_DIR, "StopSignImage.png"))
    TrafficCone = ("TrafficConeAndCylinder.blend", "absperrhut", (math.pi/2, 0, 0), 10.0, None)
    Pedestrian = ("Pedestrain.blend", "BaseMesh_Man_Simple", (math.pi/2, 0, PI), .055, None)
    Dustbin = ("Dustbin.blend", "Bin_Mesh.072", (PI/2, 0, 0), 10, None)
    FireHyrant = ("TrafficAssets.blend", "Circle.002", (0, 0, 0), 1.5, None)
    SmallPole = ("TrafficAssets.blend", "Cylinder.001", (0, 0, 0), 1.0, None) # Probably f, or a chain gate or something, probably won't use   
    SpeedLimitSign = ("SpeedLimitSign.blend", "sign_25mph_sign_25mph", (0, 0, 0), 4.0, None)
    TrafficLight = ("TrafficSignalRed.blend", "Traffic_signal1", (PI/2, 0, PI/2), 1.5, None)
    # Lane markings

    def __init__(self, file_path, obj_name, default_rotation, default_scaling, texture_path):
        self.file_path = file_path
        self.obj_name = obj_name
        self.default_rotation = default_rotation
        self.default_scaling = default_scaling
        self.texture_path = texture_path

class Asset:
    def __init__(self, asset_type: AssetType, location=None, rotation=None, scaling=None, coord_flip_correction=True):
        self.asset_type = asset_type
        self.location = location if location is not None else mathutils.Vector((0, 0, 0))
        # print("This is the location: ", self.location)
        self.rotation = rotation if rotation is not None else self.asset_type.default_rotation
        self.scaling = scaling * self.asset_type.default_scaling if scaling is not None else self.asset_type.default_scaling
        self.id = None
        self.coord_flip_correction = coord_flip_correction
        print("Created Asset of type: ", asset_type)

    def place(self, location, assets_dir = ASSETS_DIR, additional_rotation=(0, 0, 0), scaling=None):
        self.location = location

        if scaling is not None:
            self.scaling = self.scaling * scaling
        if self.coord_flip_correction:
            self.location = mathutils.Vector((-location.x, -location.z, -location.y))
        # Apply default rotation and additional rotation
        total_rotation = [d + a for d, a in zip(self.asset_type.default_rotation, additional_rotation)]
        self.rotation = mathutils.Euler(total_rotation, 'XYZ')
        
        # If it's a stop sign, translate down by 1.5 meters
        if self.asset_type == AssetType.StopSign:
            self.location.z = self.location.z - 10
        
        # Load the asset
        file_path = os.path.join(assets_dir, self.asset_type.file_path)

        # Add the object to the scene
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

            # # Add texture if it's a Stop Sign
            # if self.asset_type == AssetType.StopSign:
            #     self.apply_texture(self.asset_type.file_path)

    def apply_texture(self, texture_path):
        # Create a new material
        mat = bpy.data.materials.new(name="TextureMaterial")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get('Principled BSDF')
        texImage = mat.node_tree.nodes.new('ShaderNodeTexImage')
        texImage.image = bpy.data.images.load(texture_path)
        mat.node_tree.links.new(bsdf.inputs['Base Color'], texImage.outputs['Color'])
        self.id.data.materials.append(mat)

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

def set_output_settings(output_filepath, resolution_x=1280, resolution_y=960, frame_start=1, frame_end=1):
    scene = bpy.context.scene
    scene.render.image_settings.file_format = 'PNG'
    scene.render.filepath = output_filepath
    scene.render.resolution_x = resolution_x
    scene.render.resolution_y = resolution_y
    scene.frame_start = frame_start
    scene.frame_end = frame_end

def set_active_camera(camera_name):
    bpy.context.scene.camera = bpy.data.objects[camera_name]

def add_camera(location=(0, 0, 0), target=(0, 0, 0)):
    # Add camera
    bpy.ops.object.camera_add(location=location)
    camera = bpy.context.object

    # Point camera at target
    direction = mathutils.Vector(target) - camera.location
    rot_quat = direction.to_track_quat('-Z', 'Y')

    camera.rotation_euler = rot_quat.to_euler()

    return camera

def set_active_camera(camera_name):
    bpy.context.scene.camera = bpy.data.objects[camera_name]

def add_light(location=(0, 0, 0), light_type='POINT', energy=1000):
    # Add light
    bpy.ops.object.light_add(type=light_type, location=location)
    light = bpy.context.object

    # Set light energy
    light.data.energy = energy

    return light

def main():
    
    clear_scene()
    
    # # Create and place a stop sign
    # stop_sign = Asset(AssetType.StopSign)
    # stop_sign.place(mathutils.Vector((-2, -2, 0)), additional_rotation=(0,0,0))

    # create_all_assets()
    # create_random_cars(10)

    print("Creating assetcontroller")
    asset_controller = AssetController('data2.json')
    asset_controller.place_first_frame()
        
    save_scene(os.path.join(ASSETS_DIR, "..", "script_test.blend"))
    print("Finished creating, now rendering")

    cam = add_camera((0, 10, 2), (0, 0, 2))
    print("Camera added: ", cam)
    set_active_camera(cam.name)
    print("Camera set to active: ", cam)

    add_light((0, 0, 100), 'SUN', 100)

    set_output_settings(os.path.join(BASE_PATH, "out1.png"), frame_start=1, frame_end=1)
    print("Output settings set")
    bpy.ops.render.render(write_still=True)
    print("Rendered image")

if __name__ == "__main__":
    main()