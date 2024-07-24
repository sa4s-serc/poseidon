from .utils import *
from .neptune_step1 import *


class NeptuneStep2Base(NeptuneStepBase):
    def __init__(self, mode=str, soften_step1_sol=1.3, **kwargs):
        super().__init__(**kwargs)
        self.mode = mode
        assert mode in ["delete", "create"]
        self.moved_from, self.moved_to = {}, {}
        self.soften_step1_sol = soften_step1_sol

    def init_vars(self):
        super().init_vars()
        init_moved_from(self.data, self.solver, self.moved_from)
        init_moved_to(self.data, self.solver, self.moved_to)
        self.allocated = init_allocated(self.data, self.solver)
        self.deallocated = init_deallocated(self.data, self.solver)

    def init_constraints(self):

        self.log(self.data.old_allocations_matrix)
        self.log(self.data.core_per_req_matrix)
        self.log(self.data.workload_matrix)
        self.log(self.data.node_cores_matrix)

        super().init_constraints()
        constrain_handle_all_requests(self.data, self.solver, self.x)
        constrain_CPU_usage(self.data, self.solver, self.x)
        constrain_moved_from(self.data, self.solver, self.moved_from, self.c)
        constrain_moved_to(self.data, self.solver, self.moved_to, self.c)
        constrain_migrations(self.data, self.solver, self.c, self.allocated, self.deallocated)
        if self.mode == "delete":
            constrain_deletions(self.data, self.solver, self.c, self.allocated, self.deallocated)
        elif self.mode == "create":
            constrain_creations(self.data, self.solver, self.c, self.allocated, self.deallocated)



    def init_objective(self):
        minimize_disruption(self.data, self.objective, self.moved_from, self.moved_to, self.allocated, self.deallocated)

    def results(self):
        x, c = output_x_and_c(self.data, self.x, self.c)
        print("Step 2 - x:", x, sep='\n')
        print("Step 2 - c:", c, sep='\n')
        print("Number of delta pod deallocated")
        print(-self.deallocated.solution_value())
        print("Number of delta pod allocated")
        print(-self.allocated.solution_value())
        return x, c

class NeptuneStep2MinUtilization(NeptuneStep2Base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n = {}

    def init_vars(self):
        super().init_vars()
        init_n(self.data, self.solver, self.n)

    def init_constraints(self):
        super().init_constraints()
        constrain_n_according_to_c(self.data, self.solver, self.n, self.c)
        constrain_budget(self.data, self.solver, self.n)
        constrain_node_utilization(self.data, self.solver, self.n, self.soften_step1_sol)

    def results(self):
        x, c = super().results()
        n = output_n(self.data, self.n)
        print("Step 2 - n:", n, sep='\n')
        return x, c


class NeptuneStep2MinDelay(NeptuneStep2Base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def init_constraints(self):
        super().init_constraints()
        constrain_network_delay(self.data, self.solver, self.x, self.soften_step1_sol)


class NeptuneStep2MinDelayAndUtilization(NeptuneStep2MinUtilization):
    def __init__(self, alpha=0.5, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    def init_constraints(self):
        NeptuneStep2Base.init_constraints(self)
        constrain_n_according_to_c(self.data, self.solver, self.n, self.c)
        constrain_budget(self.data, self.solver, self.n)
        constrain_score(self.data, self.solver, self.x, self.n, self.alpha, self.soften_step1_sol)
