"""Wait for the frozen reference run, then execute the declared next protocol."""
import json
from pathlib import Path
import runpy
import sys
import time

project = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project))
root = Path(__file__).resolve().parent
parent = project/'outputs/inspection_research/residual_dispatch/run.json'
deadline = time.monotonic()+7200
(root/'launcher.json').write_text(json.dumps(dict(status='waiting_for_reference', parent=str(parent))))
while json.loads(parent.read_text())['status'] != 'complete':
    if time.monotonic() >= deadline:
        raise RuntimeError('Reference study did not complete within two hours; inspect its logs before resuming')
    time.sleep(10)
(root/'launcher.json').write_text(json.dumps(dict(status='started', protocol='65536 physical steps,3 seeds,50 test sequences')))
sys.argv = ['factory_rl.event_dispatch', '--physical-steps', '65536', '--seeds', '3',
            '--episodes', '50', '--validation', '10', '--output', 'outputs/inspection_research/event_dispatch']
runpy.run_module('factory_rl.event_dispatch', run_name='__main__')
(root/'launcher.json').write_text(json.dumps(dict(status='complete')))
