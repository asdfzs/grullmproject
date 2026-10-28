"""
List CARLA prop blueprints that could be road debris, with their real size (spawned briefly to measure).
Run with CARLA open (any map):   venv\\Scripts\\python.exe tools\\list_props.py
Output: tools\\props_sizes.txt
"""
import os, carla

KEYS = ['box', 'bag', 'garbage', 'trash', 'debris', 'barrel', 'tyre', 'tire', 'wheel', 'plank', 'tile', 'creased',
        'cardboard', 'bin', 'chair', 'table', 'trolley', 'mattress', 'gnome', 'pot', 'bucket', 'cone', 'wood', 'brick']
client = carla.Client('localhost', 2000); client.set_timeout(30)
world = client.get_world()
bps = [b for b in world.get_blueprint_library().filter('static.prop.*') if any(k in b.id for k in KEYS)]
sp = world.get_map().get_spawn_points()[0]
loc = sp.location + carla.Location(z=50)            # spawn high above the road, measure, destroy
lines = []
for b in sorted(bps, key=lambda b: b.id):
    try:
        a = world.spawn_actor(b, carla.Transform(loc, sp.rotation))
        world.wait_for_tick()
        e = a.bounding_box.extent
        lines.append(f'{b.id:40s} L {2*e.x:5.2f}m  W {2*e.y:5.2f}m  H {2*e.z:5.2f}m')
        a.destroy()
    except Exception as ex:
        lines.append(f'{b.id:40s} (could not spawn: {ex})')
txt = '\n'.join(lines); print(txt)
open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'props_sizes.txt'), 'w').write(txt + '\n')
