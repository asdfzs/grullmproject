"""
Find straight, junction-free roads in the loaded CARLA map where our lane has a SAME-direction lane on BOTH
sides (= we drive in a middle lane), and write lane-closure routes with an arrow board pointing left / right.
Both neighbour lanes are free, so only the arrow tells which way to go.

CARLA must be running (make_closure_routes.ps1 starts it with Town05):
    venv\\Scripts\\python.exe tools\\make_closure_routes.py --n 2
Writes leaderboard\\data\\bench2drive_split\\bench2drive_m<k>_closure_left.xml / _right.xml
"""
import argparse, os, random
import carla

DRIVE = carla.LaneType.Driving


def same(wp, nb):
    return nb is not None and nb.lane_type == DRIVE and nb.lane_id * wp.lane_id > 0


def middle(wp):
    return same(wp, wp.get_left_lane()) and same(wp, wp.get_right_lane())


def walk(wp, dist, step=2.0):
    pts, cur, d = [wp], wp, 0.0
    while d < dist:
        nxt = cur.next(step)
        if not nxt:
            return None
        cur = nxt[0]
        if cur.is_junction:
            return None
        pts.append(cur)
        d += step
    return pts


def free_behind(wp, dist, step=2.0):
    cur, d = wp, 0.0
    while d < dist:
        prev = cur.previous(step)
        if not prev or prev[0].is_junction:
            return False
        cur = prev[0]
        d += step
    return True


def yaw_span(pts):
    y0 = pts[0].transform.rotation.yaw
    return max(abs((p.transform.rotation.yaw - y0 + 180) % 360 - 180) for p in pts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=2, help='number of spots')
    ap.add_argument('--length', type=float, default=150.0)
    ap.add_argument('--seed', type=int, default=3)
    ap.add_argument('--out', default=os.path.join(os.environ.get('WORK_DIR', r'C:\carla_garage'),
                                                  'leaderboard', 'data', 'bench2drive_split'))
    a = ap.parse_args()
    rng = random.Random(a.seed)
    client = carla.Client('localhost', 2000)
    client.set_timeout(60)
    m = client.get_world().get_map()
    town = m.name.split('/')[-1]
    print('map:', town)
    from agents.navigation.global_route_planner import GlobalRoutePlanner
    grp = GlobalRoutePlanner(m, 1.0)

    cands = [w for w in m.generate_waypoints(4.0) if w.lane_type == DRIVE and not w.is_junction and middle(w)]
    print(f'{len(cands)} middle-lane waypoints')
    rng.shuffle(cands)
    spots, taken = [], []
    for wp in cands:
        if len(spots) >= a.n:
            break
        seg = walk(wp, a.length)
        if seg is None or yaw_span(seg) > 12 or not all(middle(p) for p in seg) or not free_behind(wp, 60.0):
            continue
        if any(seg[0].transform.location.distance(t) < 200 for t in taken):
            continue
        sparse = [seg[0], seg[5], seg[len(seg) // 2], seg[-1]]
        try:   # the leaderboard planner must keep the same road + lane, never enter a junction and not detour
            tot, prev, bad = 0.0, None, False
            # the leaderboard snaps the XML positions (rounded to 0.1 m) with map.get_waypoint -> check exactly that
            # (XML keeps the real height: with z=0 a point on a bridge snaps to the road BELOW it -> long detour)
            snapped = [m.get_waypoint(carla.Location(x=round(p.transform.location.x, 1),
                                                     y=round(p.transform.location.y, 1),
                                                     z=round(p.transform.location.z, 1))) for p in sparse]
            if any(sp.road_id != wp.road_id or sp.lane_id != wp.lane_id for sp in snapped):
                continue
            for p0, p1 in zip(snapped[:-1], snapped[1:]):
                for w, _ in grp.trace_route(p0.transform.location, p1.transform.location):
                    if w.is_junction or w.road_id != wp.road_id:
                        bad = True
                    if prev is not None:
                        tot += w.transform.location.distance(prev)
                    prev = w.transform.location
            if bad or tot > 1.2 * a.length:
                continue
        except Exception:
            continue
        spots.append(sparse)
        taken.append(seg[0].transform.location)

    weather = ('cloudiness="5.0" fog_density="0.0" precipitation="0.0" precipitation_deposits="0.0" '
               'sun_altitude_angle="70.0" sun_azimuth_angle="0.0" wetness="0.0" wind_intensity="5.0"')
    ids = []
    for k, pts in enumerate(spots, start=1):
        trig = pts[1].transform
        pos = '\n'.join(f'         <position x="{p.transform.location.x:.1f}" y="{p.transform.location.y:.1f}" '
                        f'z="{p.transform.location.z:.1f}"/>' for p in pts)
        for d in ('left', 'right'):
            xml = f'''<routes><route id="8{k:02d}{0 if d == 'left' else 1}" town="{town}">
      <waypoints>
{pos}
      </waypoints>
      <scenarios>
         <scenario name="NarrowCorridor_1" type="NarrowCorridor">
            <distance value="45"/>
            <length value="24"/>
            <layout value="closure"/>
            <sign value="static.prop.trafficwarning"/>
            <sign_yaw value="270"/>
            <sign_lat value="0.0"/>
            <sign_before value="-3"/>
            <sign_arrow value="{d}"/>
            <trigger_point x="{trig.location.x:.1f}" y="{trig.location.y:.1f}" yaw="{trig.rotation.yaw:.1f}" z="{trig.location.z:.1f}"/>
         </scenario>
      </scenarios>
      <weathers>
         <weather {weather} route_percentage="0"/>
         <weather {weather} route_percentage="100"/>
      </weathers>
   </route>
   </routes>
'''
            rid = f'm{k}_closure_{d}'
            open(os.path.join(a.out, f'bench2drive_{rid}.xml'), 'w', encoding='utf-8').write(xml)
            ids.append(rid)
        p0 = pts[0]
        print(f'  m{k}: start ({p0.transform.location.x:.1f},{p0.transform.location.y:.1f}, z {p0.transform.location.z:.1f}) road {p0.road_id} '
              f'lane {p0.lane_id}, lane width {p0.lane_width:.2f} m')
    if len(spots) < a.n:
        print(f'!! only {len(spots)} spot(s) found (straight middle lane, {a.length:.0f} m, no junction)')
    print('\nNext:\n  .\\run_intervene.ps1 ' + ','.join(ids) + ' -Llm gemini -NoTraffic -LlmInput photo')


if __name__ == '__main__':
    main()
