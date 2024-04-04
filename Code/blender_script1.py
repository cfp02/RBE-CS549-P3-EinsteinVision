import os
import math
import sys
import enum
import random
import time
import json

try:
    import bpy # type: ignore
    import mathutils # type: ignore
    BASE_PATH = os.path.dirname(bpy.data.filepath)
except ImportError:
    print("ImportError: This script must be run from within Blender. Functionality will be limited.")
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))

PI = math.pi
ASSETS_DIR = os.path.normpath(os.path.join(BASE_PATH, "../P3Data", "Assets"))
OUTPUT_PATH = os.path.normpath(os.path.join(BASE_PATH, "../Output"))
JSON_DATA_PATH = os.path.normpath(os.path.join(BASE_PATH,"../JSONData"))

class AssetKey(enum.Enum):
    YOLOZOE_ASSETS = 'yolozoe-assets'
    CARPOSE_ASSETS = '3dcarpose-assets'
    LANES = 'lanes'

    @property
    def coordinate_scaling(self):
        # Need to multiply by x to convert from network's arbitrary units to blender units
        match self:
            case AssetKey.YOLOZOE_ASSETS:
                return 50
            case AssetKey.CARPOSE_ASSETS:
                return .005
            case AssetKey.LANES:
                return 10
            case _:
                return 1
    
    def coordinate_flip(self, location:tuple):
        # Need to change coordinate directions for some networks
        # Location is in the form (x, y, z)
        x, y, z = location
        match self:
            case AssetKey.YOLOZOE_ASSETS:
                return (-x, -z, -y)  #(-location[0], -location[2], -location[1])
            case AssetKey.CARPOSE_ASSETS:
                return (x, -1/z, y)
            case AssetKey.LANES:
                return (-x, -z, -y)
            case _:
                return location
            
    def rotation_flip(self, rotation):
        # Need to change coordinate directions for some networks
        # Rotation is in the form (x, y, z)
        match self:
            case AssetKey.YOLOZOE_ASSETS:
                return rotation
            case AssetKey.CARPOSE_ASSETS:
                return rotation #(rotation[0], rotation[1], -rotation[2])
            case AssetKey.LANES:
                return rotation
            case _:
                return rotation

class AssetController:
    '''
    In charge of holding all of the Assets in the scene and managing actions related to them.
    Holds dict of frames. Each frame contains a list of assets (created from yolo and zoedepth), a list of lanes, and assets from 3dboundingbox network which shows poses of objects
    '''

    def __init__(self, json_files: list[tuple[AssetKey, str]] = None, scene = 1):
        # json_files is a list of tuples corresponding to the key and the json file path. The json file path is assumed to already be an absolute path

        self.scene = scene # Which scene we are looking at
        self.frames : dict[int, dict[str, list[Asset]]] = {
        }
        # frame: {
        #     AssetKey.YOLOZOE_ASSETS: [],
        #     AssetKey.POSE_ASSETS: [],
        #     AssetKey.LANES: []
        # }

        if json_files is not None:
            # Load all the json files into the dictionaries holding the assets
            for key, json_file in json_files:
                json_file = os.path.normpath(json_file)
                if not os.path.exists(json_file):
                    print("Error: JSON file does not exist: ", json_file, "\n Is it an absolute path?")
                    continue

                self.json_to_assets(json_file, key)

    def add_lanes(self, lanes_json_file, frame=0):
        lanes = []
        
        with open(lanes_json_file, 'r') as file:
            data = json.load(file)
            lanes = data['lanes']
    
    # Converts a json file to a list of assets, adds them to the AssetController frames
    def json_to_assets(self, json_file, asset_key: AssetKey):
        json_reader = JSONReader(json_file)
        print('Size of json data: ', len(json_reader.data))
        for frame_idx, frame_data in enumerate(json_reader.data):
            frame = frame_data['frame'] # Get the frame number
            # assets = frame_data['assets']
            assets = frame_data['objects']  # Get the list of assets
            asset_list = []
            for asset_json in assets:
                # Create the asset object, might have to change for each AssetKey/network
                # loc = mathutils.Vector([coord * 50 for coord in asset_json['location']])
                # rot = mathutils.Euler(asset_json['rotation'], 'XYZ')
                loc = tuple([coord * asset_key.coordinate_scaling for coord in asset_json['location']])
                rot = tuple(asset_json['rotation'])
                asset = Asset(self.asset_type_from_string(asset_json['type']), 
                              location=loc,
                              rotation=rot,
                              scaling=asset_json['scaling'])
                asset_list.append(asset)
            self.load_assets(asset_list, asset_key, frame)

    # Add a list of assets to a specific frame under a specific key (from a specific network)
    def load_assets(self, assets, asset_key: AssetKey, frame:int):
        if frame not in self.frames:
            self.frames[frame] = {}
        if asset_key not in self.frames[frame]:
            self.frames[frame][asset_key] = []
        self.frames[frame][asset_key].extend(assets)
        print("Added", len(assets), "assets to frame: ", frame, 'with key:', asset_key)

    def place_first_frame(self, which_assets:AssetKey):
        print(min(self.frames.keys()))
        self.place_assets(frame = min(self.frames.keys()), asset_key=which_assets)

    # Place all assets from in a specific frame in Blender
    def place_assets(self, frame:int, asset_key:AssetKey):
        if frame in self.frames:
            if asset_key in self.frames[frame]:
                for asset in self.frames[frame][asset_key]:
                    asset.place(asset.location, additional_rotation=asset.rotation, asset_key=asset_key)
        else:
            print("No assets found for frame: ", frame)

    # Determines which AssetType to use based on the string used in the input json file
    def asset_type_from_string(self, type_str):
        asset_type = None
        # Phase 1 includes only lanes, vehicles, pedestrians, traffic lights, stop signs
        match type_str:
            case "Sedan" | 'car':
                asset_type = AssetType.Sedan
            case "SUV":
                asset_type = AssetType.SUV
            case "PickupTruck":
                asset_type = AssetType.PickupTruck
            case "Truck" | 'truck':
                asset_type = AssetType.Truck
            case "Bicycle" | 'bicycle':
                asset_type = AssetType.Bicycle
            case "Motorcycle" | 'motorcycle':
                asset_type = AssetType.Motorcycle
            case "StopSign" | "stop sign":
                # asset_type = AssetType.StopSign
                asset_type = AssetType.SpeedLimitSign
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
    # Asset types: (file_path, obj_name, default_rotation, default_translation, default_scaling, texture_path=None)
    Sedan = ("Vehicles/SedanAndHatchback.blend", "Car", (0, 0, 0), (0,0,0), .12, None)
    SUV = ("Vehicles/SUV.blend", "Jeep_3_", (0, 0, PI), (0, 0, 0), 20, None)
    PickupTruck = ("Vehicles/PickupTruck.blend", "PickupTruck", (PI/2, 0, PI/2), (0,0,0), 5, None)
    Truck = ("Vehicles/Truck.blend", "Truck", (0, 0, PI), (0,0,0), .004, None)
    Bicycle = ("Vehicles/Bicycle.blend", "roadbike 2.0.1", (PI/2, 0, PI), (0,0,0), 0.9, None)
    Motorcycle = ("Vehicles/Motorcycle.blend", "B_Wheel", (PI/2, 0, -PI/2), (0,0,0), .04, None)

    StopSign = ("StopSign.blend", "StopSign_Geo", (PI/2, 0, PI/2), (0,0,-10), 2.0, os.path.join(ASSETS_DIR, "StopSignImage.png"))
    TrafficCone = ("TrafficConeAndCylinder.blend", "absperrhut", (PI/2, 0, 0), (0,0,0), 10.0, None)
    Pedestrian = ("Pedestrain.blend", "BaseMesh_Man_Simple", (PI/2, 0, PI), (0,0,0), .055, None)
    Dustbin = ("Dustbin.blend", "Bin_Mesh.072", (PI/2, 0, 0), (0,0,0), 10, None)
    FireHyrant = ("TrafficAssets.blend", "Circle.002", (0, 0, 0), (0,0,0), 1.5, None)
    SmallPole = ("TrafficAssets.blend", "Cylinder.001", (0, 0, 0), (0,0,0), 1.0, None) # Probably f, or a chain gate or something, probably won't use   
    SpeedLimitSign = ("SpeedLimitSign.blend", "sign_25mph_sign_25mph", (0, 0, PI), (0,0,-10), 4.0, os.path.join(ASSETS_DIR, "SpeedLimitSigns/Speed_Limit_20.png"))
    TrafficLight = ("TrafficSignalRed.blend", "Traffic_signal1", (PI/2, 0, PI/2), (0,0,0), 1.5, None)
    # Lane markings

    def __init__(self, file_path, obj_name, default_rotation, default_translation, default_scaling, texture_path):
        self.file_path = file_path
        self.obj_name = obj_name
        self.default_rotation = default_rotation
        self.default_translation = default_translation
        self.default_scaling = default_scaling
        self.texture_path = texture_path

class Asset:
    def __init__(self, asset_type: AssetType, location=None, rotation=None, scaling=None, coord_flip_correction=True):
        self.asset_type = asset_type
        self.location = location if location is not None else (0,0,0) # mathutils.Vector((0, 0, 0))
        # print("This is the location: ", self.location)
        self.rotation = rotation if rotation is not None else (0,0,0)
        self.scaling = scaling * self.asset_type.default_scaling if scaling is not None else self.asset_type.default_scaling
        self.id = None
        self.coord_flip_correction = coord_flip_correction
        print("Created Asset of type: ", asset_type)

    def place(self, location, assets_dir = ASSETS_DIR, additional_rotation=(0, 0, 0), scaling=None, asset_key:AssetKey=None):
        self.location = location

        if scaling is not None:
            self.scaling = self.scaling * scaling


        # TODO: Apply coordinate flip and rotations when adding them instead of here
        self.location = mathutils.Vector(asset_key.coordinate_flip(location))
        # additional_rotation = AssetKey.YOLOZOE_ASSETS.rotation_flip(additional_rotation)
        # Apply default rotation and additional rotation
        total_rotation = [d + a for d, a in zip(self.asset_type.default_rotation, additional_rotation)]
        self.rotation = mathutils.Euler(total_rotation, 'XYZ')
        
        # # If it's a stop sign, translate down by 1.5 meters
        # if self.asset_type == AssetType.StopSign:
        #     self.location.z = self.location.z - 10

        # Apply default translation
        self.location = self.location + mathutils.Vector(self.asset_type.default_translation)
        
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

            # Add texture if it's a Speed Limit Sign
            if self.asset_type == AssetType.SpeedLimitSign:
                self.change_speed_limit_texture(appended_obj, 40)

    def change_speed_limit_texture(self, obj, speed_limit):
        speed_lims = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]
        if speed_limit not in speed_lims:
            print("Speed limit not in list of valid speed limits")
            return
        speed_lim_str = str(speed_limit)
        speed_lim_png_path = self.asset_type.texture_path
        # Replace the -6:-4 with the speed limit string
        new_texture_path = speed_lim_png_path[:-6] + speed_lim_str + speed_lim_png_path[-4:]

        mat = obj.data.materials[1] # second material
        if not mat:
            print("No material found")
            return
        if not mat.use_nodes:
            mat.use_nodes = True

        nodes = mat.node_tree.nodes
        img_tex_node = next((node for node in nodes if node.type == 'TEX_IMAGE'), None)

        if img_tex_node:
            new_img = bpy.data.images.load(new_texture_path, check_existing=True)
            img_tex_node.image = new_img
        else:
            print("Image Texture node not found.")

        

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

def create_traffic_lane(points: list, which_lane=0):
    curve_data = bpy.data.curves.new(name='lane' + str(which_lane), type='CURVE')
    curve_data.dimensions = '3D'

    # Create a spline
    spline = curve_data.splines.new('BEZIER')
    spline.bezier_points.add(len(points) - 1)

    for i, point in enumerate(points):
        spline.bezier_points[i].co = point
        spline.bezier_points[i].handle_left_type = 'AUTO'
        spline.bezier_points[i].handle_right_type = 'AUTO'

    curve_obj = bpy.data.objects.new('lane_object' + str(which_lane), curve_data)
    bpy.context.collection.objects.link(curve_obj)

    bpy.context.view_layer.objects.active = curve_obj
    curve_obj.select_set(True)

    # Make the curve a mesh
    bpy.ops.object.convert(target='MESH')

    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    #Thicken
    bpy.ops.mesh.extrude_region_move(TRANSFORM_OT_translate={"value":(1, 0, 0)})
    bpy.ops.object.mode_set(mode='OBJECT')

    return curve_obj


def create_and_apply_texture_material(lane_object, texture_path):
    # Create a new material
    mat = bpy.data.materials.new(name="LaneMaterial")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get('Principled BSDF')

    # Create an Image Texture node and load the texture
    tex_image = mat.node_tree.nodes.new('ShaderNodeTexImage')
    tex_image.image = bpy.data.images.load(texture_path)

    # Connect the Image Texture node to the Base Color of the Principled BSDF
    mat.node_tree.links.new(bsdf.inputs['Base Color'], tex_image.outputs['Color'])

    mat.node_tree.links.new(bsdf.inputs['Alpha'], tex_image.outputs['Alpha'])
    mat.blend_method = 'BLEND' 

    # Assign the material to the object
    if len(lane_object.data.materials):
        lane_object.data.materials[0] = mat  # Replace the first material
    else:
        lane_object.data.materials.append(mat)  # Add new material

def read_lane_data(json_file):
    # Returns list of lanes which is a list of points
    # Each lane is a list of points in the form [x, y, z] but should be list[tuple[float, float, float]]
    a = AssetKey.LANES
    with open(json_file, 'r') as file:
        data = json.load(file)[0]
        print("Data: ", data)
        json_lanes = data['lanes']

    lanes = []
    for l in json_lanes:
        lane = []
        for point in l:
            point = ((a.coordinate_scaling *p) for p in point)
            point = a.coordinate_flip(point)
            point = tuple(point)
            lane.append(point)
        lanes.append(list(lane))
    return lanes
    
def create_traffic_lanes(lanes_data):
    for i, lane_data in enumerate(lanes_data):
        points = [mathutils.Vector(point) for point in lane_data]
        lane = create_traffic_lane(points, i)
        # create_and_apply_texture_material(lane, lane_data['texture_path'])

def main():

    print("Creating assetcontroller")
    asset_controller = AssetController(scene = 1, json_files= [
        (AssetKey.YOLOZOE_ASSETS, os.path.join(JSON_DATA_PATH, 'scene1/scene1-yolodepth2140.json')),
        # (AssetKey.CARPOSE_ASSETS, os.path.join(JSON_DATA_PATH, 'scene1/scene1-carposes2140.json')),
        # (AssetKey.CARPOSE_ASSETS, os.path.join(JSON_DATA_PATH, 'scene5/scene5-carposes170.json')),
    ])

    clear_scene()

    asset_controller.place_first_frame(AssetKey.YOLOZOE_ASSETS)
        
    # save_scene(os.path.join(ASSETS_DIR, "..", "script_test.blend"))
    print("Finished creating, now rendering")

    cam = add_camera((0, 10, 2), (0, 0, 2))
    print("Camera added: ", cam)
    set_active_camera(cam.name)
    print("Camera set to active: ", cam)

    add_light((0, 0, 100), 'SUN', 100)
    add_light((0, 0, 0), 'SUN', 40)

    # points = [(0, 0, 0), (0, -2, 0), (1, -8, 0), (2, -30, 0), (3, -40, 1), (4, -50, 0)]
    # lane = create_traffic_lane([mathutils.Vector(point) for point in points], 0)
    # create_and_apply_texture_material(lane, os.path.join(ASSETS_DIR, "DashedLine.png"))

    create_traffic_lanes(read_lane_data(os.path.join(JSON_DATA_PATH, "scene1", "scene1-lanes2140.json")))

    set_output_settings(os.path.join(OUTPUT_PATH, "P2Scene1Last.png"), frame_start=1, frame_end=1)
    print("Output settings set")
    bpy.ops.render.render(write_still=True)
    print("Rendered image")

if __name__ == "__main__":
    main()