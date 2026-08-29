import tempfile,os
from factory.drive import DrivePaths
from factory.config import Config

def main():
 with tempfile.TemporaryDirectory() as t:
  p=DrivePaths(t); p.ensure_tree(); c=Config.load(t)
  assert c.target_video_seconds==90
  assert os.path.isdir(p('04_AUDIO_LIBRARY','music'))
  print('basic architecture self-test passed')
if __name__=='__main__': main()
