import util
import math
import collections
from engine.const import Const
from util import Belief

def observe(self, agentX: int, agentY: int, observedDist: float) -> None:
    # BEGIN_YOUR_CODE (our solution is 9 lines of code, but don't worry if you deviate from this)
        belief = self.belief
        for r in range(belief.getNumRows()):
            for c in range(belief.getNumCols()):
                x, y = util.colToX(c), util.rowToY(r)
                dist = math.sqrt((x-agentX)**2 + (y-agentY)**2)
                prob = util.pdf(dist, Const.SONAR_STD, observedDist)
                belief.setProb(r, c, belief.getProb(r, c)*prob)
        belief.normalize()

    # END_YOUR_CODE

def elapseTime(self) -> None:
    if self.skipElapse: ### ONLY FOR THE GRADER TO USE IN Problem 1
        return
    # BEGIN_YOUR_CODE (our solution is 8 lines of code, but don't worry if you deviate from this)
    belief = self.belief
    new_belief = util.Belief(belief.getNumRows(), belief.getNumCols(), 0.0)      
        
    for (oldTile, newTile), transProb in self.transProb.items():           
        new_belief.addProb(newTile[0], newTile[1], belief.getProb(oldTile[0], oldTile[1])  * transProb)

    new_belief.normalize()   
    self.belief = new_belief
    # END_YOUR_CODE

########################################################################################################

def elapseTime(self) -> None:
        # BEGIN_YOUR_CODE (our solution is 6 lines of code, but don't worry if you deviate from this)
        new_particles = collections.defaultdict(int)
        for particle in self.particles:
            for _ in range(self.particles(particle)):
                new_Tile = util.weightedRandomChoice(self.transProbDict[particle])
                new_particles[new_Tile] += 1
        self.particle = new_particles            
        # END_YOUR_CODE

def observe(self, agentX: int, agentY: int, observedDist: float) -> None:
        # BEGIN_YOUR_CODE (our solution is 13 lines of code, but don't worry if you deviate from this)
        # re-weight
        weights = collections.defaultdict(float)
        for Tile in self.particles:
            x, y = util.colToX(Tile[1]), util.rowToY(Tile[0])
            dist = math.sqrt((agentX-x**2)+(agentY-y)**2)
            prob = util.pdf(dist, Const.SONAR_STD, observedDist)
            weights[Tile] = prob * self.particles[Tile]

        if sum(weights.values()) == 0:
            return
        
        # re_sample
        new_particles = collections.defaultdict(int)
        for _ in range(self.NUM_PARTICLES):
            new_Tile = util.weightedRandomChoice(weights)
            new_particles[new_Tile] += 1
        self.particles = new_particles
        # END_YOUR_CODE

        self.updateBelief()


