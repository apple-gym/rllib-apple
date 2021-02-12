Trying ray

Speed is essentially the same despite multiple workers. 2 steps/s, 0.5 updates /s. Can't use fp16 but it's the same speed anyway


Hopefully their alg is more mature...

I can also try having multiple trainers?

# Speed

How? in tensorboard get `num_steps_sampled` `num_steps_trained` divided by time

- initial: 2 steps/s, 0.5 updates /s.
- 3 workers, 1 trainer: 
  - stepped, 100k 4hours: 7s/s
  - trained: 0.69/s
- single threaded
  - trained: 0.9/s (about the same)
  - stepped, 1.7/s (less)

# Notes

- how to work out speed? num_steps_sampled num_steps_trained
- best way to get speed? Looks like 1 trainer, and multiple workers. But for mem I found 3 is good, and a smaller mem buffer
- am I doing the model right?
