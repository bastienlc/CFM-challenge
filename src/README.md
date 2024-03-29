# CFM-challenge : Stock Classification from High Frequency Market Data

This repository contains our work for the [CFM challenge](https://challengedata.ens.fr/participants/challenges/146/) of the "Learning and Generation through Random Sampling" course at Collège de France.

The goal of this challenge is to predict the stock corresponding to a snapshot of a given order book. Each sample is chronological sequence of 100 events of orders, posted or traded, for a given stock. To make this task challenging, a lot of the usual data features have been removed by the organizers, and some particularly revealing properties, like price or best bid and ask, have been hidden by centering the data around the first event of each sample.

You can find our detailed report [here](./report/report.pdf).

## Contributors

[@bastienlc](https://github.com/bastienlc),
[@s89ne](https://github.com/s89ne)
