from evsim.common import(DT_H,EPS,OFF,ON,M_UNMET,GAMMA,EVParams)
from functools import lru_cache
import numpy as np
def step_soc(ev:EVParams,soc: float,action: int)-> float:
    #return the EV's next stte of charge after one action
    if action not in (OFF,ON):
        raise ValueError("action must be OFF(0) or ON(1)")
    if action==OFF:
        return soc
    next_soc=(soc+ev.eta*ev.p_rated_kw*DT_H/ev.e_cap_kwh)
    return min(1.0,next_soc)
import math

def k_req(ev:EVParams,soc:float)-> int:
    #return the min no.of ON slots needed to reach the target 
    deficit=max(0.0,ev.soc_target-soc)
    if deficit<=EPS:
        return 0
    energy_needed=deficit*ev.e_cap_kwh
    energy_per_slot=ev.eta*ev.p_rated_kw*DT_H
    if energy_per_slot<=0:
        raise ValueError("Charging energy per slot must be positive")
    return math.ceil(energy_needed/energy_per_slot -EPS)

def solve_q(
    ev: EVParams,
    soc: float,
    t: int,
    price: np.ndarray,
) -> tuple[float, float]:
    'return (Q_OFF,Q_ON) using backward dynamic programming'
    if not 0.0 <=soc<=1.0:
        raise ValueError("soc must be between 0 and 1")
    if t<ev.arrive_slot or t>=ev.depart_slot:
        raise ValueError("t must be an active EV time slot")
    if len(price)<ev.depart_slot:
        raise ValueError("price array is too short")
    if not np.all(np.isfinite(price)):
        raise ValueError("price must contain finite values")
    @lru_cache(maxsize=None)
    def value(current_soc:float,current_t:int)->float:
        #terminal condition : EV has departed
        if current_t >= ev.depart_slot:
            unmet_kwh=max(0.0,ev.soc_target-current_soc)*ev.e_cap_kwh
            return M_UNMET*unmet_kwh
        #calculate cost for OFF
        soc_off=step_soc(ev,current_soc,OFF)
        cost_off=(GAMMA * value(soc_off,current_t+1))
        #calculate cost for ON
        soc_on=step_soc(ev,current_soc,ON)
        energy_cost=(ev.p_rated_kw*DT_H* price[current_t])
        cost_on=(energy_cost+GAMMA*value(soc_on,current_t+1))
        return min(cost_off,cost_on)
    #calculate both action values at the current state
    soc_off=step_soc(ev,soc,OFF)
    q_off=GAMMA*value(soc_off,t+1)
    soc_on=step_soc(ev,soc,ON)
    energy_cost=ev.p_rated_kw*DT_H*price[t]
    q_on=(energy_cost+GAMMA* value(soc_on,t+1))
    return q_off,q_on