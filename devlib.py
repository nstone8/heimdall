from heimdall.device import RidgeSpec as Ridge
from heimdall.device import RidgeSpecSemiGutter as SemiGutter
import math
dev_a=Ridge(ridge_height=400,ridge_width=40,ridge_spacing=240,ridge_angle=math.pi/6)
test=SemiGutter(ridge_height=400,ridge_width=40,ridge_spacing=240,ridge_angle=math.pi/6)
