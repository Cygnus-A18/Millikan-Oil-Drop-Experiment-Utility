import numpy as np

class Trial:
    def __init__(self, rise_times, fall_times, ionized_rise_times, ionized_fall_times, Date, R):
        self.rise_times = rise_times
        self.fall_times = fall_times
        self.ionized_rise_times = ionized_rise_times
        self.ionized_fall_times = ionized_fall_times
        self.date = Date
        self.R = R

        self.all_rise_times = [rise_times]
        for time in self.ionized_rise_times:
            self.all_rise_times.append(time)
        
        self.all_fall_times = [fall_times]
        for time in self.ionized_fall_times:
            self.all_fall_times.append(time)

        self.average_rise_times = [self._get_weighted_average(times) for times in self.all_rise_times]
        self.average_fall_times = [self._get_weighted_average(times) for times in self.all_fall_times]    

    def __getitem__(self, index):
        if isinstance(index, tuple):
            index, ionization_index = index
        else:
            ionization_index = 0

        if (type(index) == int):
            try: 
                return (self.all_rise_times[ionization_index][index], self.all_fall_times[ionization_index][index])
            except IndexError:
                r = self.all_rise_times[ionization_index][index] if index < len(self.all_rise_times[ionization_index]) else None
                f = self.all_fall_times[ionization_index][index] if index < len(self.all_fall_times[ionization_index]) else None
                return (r, f)       

        elif (index == "rise_time"):
            return self.average_rise_times[ionization_index]
        elif (index == "rise_times"):
            return self.all_rise_times[ionization_index]
        elif (index == "fall_time"):
            return self.average_fall_times[ionization_index]
        elif (index == "fall_times"):
            return self.all_fall_times[ionization_index]

        else:
            raise IndexError(f"invalid index: {index}")
    
    def get_all_q(self):
        qs = []
        for i in range(len(self.all_rise_times)):
            qs.append(self.get_q(i))

        return qs

    def get_all_r(self):
        rs = []
        for i in range(len(self.all_rise_times)):
            rs.append(self.get_r(i))

        return rs

    def get_all_n(self):
        ns = []
        for i in range(len(self.all_rise_times)):
            ns.append(self.get_n(i))

        return ns

class DropletData:
    def __init__(self):
        self.trials = []

    def __getitem__(self, index):
        if type(index) == int:
            return self.trials[index]
        elif index == "fall_times":
            return [t["fall_time"] for t in self.trials]
        elif index == "rise_times":
            return [t["rise_time"] for t in self.trials]

    def add_trial_from_data(self, rise_times, fall_times, Date, R, ionized_rise_times=[], ionized_fall_times=[]):
        trial = Trial(rise_times, fall_times, ionized_rise_times, ionized_fall_times, Date, R)
        self.trials.append(trial)

    def add_trial(self, trial):
        self.trials.append(trial)

def get_weighted_average(values):
        if len(values) == 0:
            return np.nan
        values = np.array(values)
        median = np.median(values)
        rlist = values - median
        MAD = np.median(abs(rlist))
        if MAD == 0:
            return np.mean(values)
        weights = 1 / (1 + (rlist / MAD)**2)

        return np.average(values, weights=weights)