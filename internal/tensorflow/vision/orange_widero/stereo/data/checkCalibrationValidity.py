import argparse
from os.path import join, exists
from mdblib.utils.clip_to_location import getClipLocation
from mdblib.utils.db_conn import get_conn


def checkValidity(clips):
    table_name = 'BETA_session'
    conn = get_conn()
    clip2sess = {}
    for clipName in clips:
        if clipName.endswith('.pls') and '/mobileye' in clipName:
            with open(clipName) as p:
                for l in p:
                    if 'File' in l:
                        clip = l.split('\n')[0].split('=')[1]
                        clip_path = getClipLocation(clip)
                        sess_path = "/".join(clip_path.split("/")[:-3])
                        break
        else:
            clip_path = getClipLocation(clipName)
            sess_path = "/".join(clip_path.split("/")[:-3])
        clip2sess[clipName] = sess_path

    sess2valid = {}
    for sess_path in list(set(clip2sess.values())):
        sess_valid = True
        for sector in ['Front', 'Rear', 'Left', 'Right']:
            try:
                c = conn.cursor()
                c.execute("SELECT {} FROM {} WHERE path = %s".format('valid_cam2cam', table_name),
                          (join(sess_path, 'Clips_' + sector),))
                res = c.fetchone()
                sess_valid = sess_valid and str(res[0]) in ['valid', 'valid_fixed']
            except:
                print('sector %s failed for sess %s, invalidate session' % (sector, sess_path))
                sess_valid = False
                break
        sess2valid[sess_path] = sess_valid

    clips2valid = {}
    for clip in clips:
        clips2valid[clip] = sess2valid[clip2sess[clip]]
    return clips2valid

def main(args):
    clips = []
    if exists(args['inp']):
        with open(args['inp']) as f:
            for l in f:
                clipName = l.split('\n')[0]
                clips.append(clipName)
    else:
        clips = [args['inp']]

    clips2valid = checkValidity(clips)

    if args['out'] == '':
        for clip in clips:
            print('%s,%s' % (clip, clips2valid[clip]))
    else:
        with open(args['out'], 'w') as f:
            for clip in clips:
                f.write('%s,%s\n' % (clip, clips2valid[clip]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='check validity of full surround calibration of clip/clip list/pls list')
    parser.add_argument('inp', help='Input clip name or clip list', default='')
    parser.add_argument('-o', '--out', help='out_file', default='')
    args = vars(parser.parse_args())
    main(args)