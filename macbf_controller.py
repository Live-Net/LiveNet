import numpy as np
import macbf.core as core
import macbf.config_macbf as config_macbf
from config import *
import tensorflow as tf


def build_evaluation_graph(num_agents):
    # s is the state vectors of the agents
    s = tf.placeholder(tf.float32, [num_agents, 4])
    # g is the goal states
    g = tf.placeholder(tf.float32, [num_agents, 2])
    # x is difference between the state of each agent and other agents
    x = tf.expand_dims(s, 1) - tf.expand_dims(s, 0)
    # h is the CBF value of shape [num_agents, TOP_K, 1], where TOP_K represents
    # the K nearest agents
    h, mask, indices = core.network_cbf(x=x, r=config_macbf.DIST_MIN_THRES)
    # a is the control action of each agent, with shape [num_agents, 3]
    a = core.network_action(s=s, g=g, obs_radius=config_macbf.OBS_RADIUS, indices=indices)
    # a_res is delta a. when a does not satisfy the CBF conditions, we want to compute
    # a a_res such that a + a_res satisfies the CBF conditions
    a_res = tf.Variable(tf.zeros_like(a), name='a_res')
    loop_count = tf.Variable(0, name='loop_count')
   
    def opt_body(a_res, loop_count):
        # a loop of updating a_res
        # compute s_next under a + a_res
        dsdt = core.dynamics(s, a + a_res)
        s_next = s + dsdt * config_macbf.TIME_STEP
        x_next = tf.expand_dims(s_next, 1) - tf.expand_dims(s_next, 0)
        h_next, mask_next, _ = core.network_cbf(
            x=x_next, r=config_macbf.DIST_MIN_THRES, indices=indices)
        # deriv should be >= 0. if not, we update a_res by gradient descent
        deriv = h_next - h + config_macbf.TIME_STEP * config_macbf.ALPHA_CBF * h
        deriv = deriv * mask * mask_next
        error = tf.reduce_sum(tf.math.maximum(-deriv, 0), axis=1)
        # compute the gradient to update a_res
        error_gradient = tf.gradients(error, a_res)[0]
        a_res = a_res - config_macbf.REFINE_LEARNING_RATE * error_gradient
        loop_count = loop_count + 1
        return a_res, loop_count

    def opt_cond(a_res, loop_count):
        # update u_res for REFINE_LOOPS
        cond = tf.less(loop_count, config_macbf.REFINE_LOOPS)
        return cond
    
    with tf.control_dependencies([
        a_res.assign(tf.zeros_like(a)), loop_count.assign(0)]):
        a_res, _ = tf.while_loop(opt_cond, opt_body, [a_res, loop_count])
        a_opt = a + a_res

    dsdt = core.dynamics(s, a_opt)
    s_next = s + dsdt * config_macbf.TIME_STEP
    x_next = tf.expand_dims(s_next, 1) - tf.expand_dims(s_next, 0)
    h_next, mask_next, _ = core.network_cbf(x=x_next, r=config_macbf.DIST_MIN_THRES, indices=indices)
    
    # compute the value of loss functions and the accuracies
    # loss_dang is for h(s) < 0, s in dangerous set
    # loss safe is for h(s) >=0, s in safe set
    # acc_dang is the accuracy that h(s) < 0, s in dangerous set is satisfied
    # acc_safe is the accuracy that h(s) >=0, s in safe set is satisfied
    (loss_dang, loss_safe, acc_dang, acc_safe) = core.loss_barrier(
        h=h_next, s=s_next, r=config_macbf.DIST_MIN_THRES, 
        ttc=config_macbf.TIME_TO_COLLISION, eps=[0, 0])
    # loss_dang_deriv is for doth(s) + alpha h(s) >=0 for s in dangerous set
    # loss_safe_deriv is for doth(s) + alpha h(s) >=0 for s in safe set
    # loss_medium_deriv is for doth(s) + alpha h(s) >=0 for s not in the dangerous
    # or the safe set
    (loss_dang_deriv, loss_safe_deriv, acc_dang_deriv, acc_safe_deriv
        ) = core.loss_derivatives(s=s_next, a=a_opt, h=h_next, x=x_next, 
        r=config_macbf.DIST_MIN_THRES, ttc=config_macbf.TIME_TO_COLLISION, alpha=config_macbf.ALPHA_CBF, indices=indices)
    # the distance between the u_opt and the nominal u
    loss_action = core.loss_actions(s, g, a, r=config_macbf.DIST_MIN_THRES, ttc=config_macbf.TIME_TO_COLLISION)

    loss_list = [loss_dang, loss_safe, loss_dang_deriv, loss_safe_deriv, loss_action]
    acc_list = [acc_dang, acc_safe, acc_dang_deriv, acc_safe_deriv]

    return s, g, a_opt, loss_list, acc_list


class macbf_controller:
    def __init__(self, model_definition_filepath, static_obs, goal):

        static_obs_macbf = []
        for obs in static_obs:
            static_obs_macbf.append([obs[0], obs[1], 0, 0])
        self.static_obs_np = np.array(static_obs_macbf)    # Shape: (Num obs, 4)
        self.static_obs_goals = np.copy(self.static_obs_np)[:, :2]
        
        self.goal_np = goal[:2]
        self.goal_np = np.expand_dims(self.goal_np, axis=0)
        self.num_agents = 2 + self.static_obs_np.shape[0]

        self.s, self.g, self.a, self.loss_list, self.acc_list = build_evaluation_graph(self.num_agents)
        model_path = 'macbf/models/model_iter_69999'
        # model_path = 'macbf/models/model_save'
        vars = tf.trainable_variables()
        vars_restore = []
        for v in vars:
            if 'action' in v.name or 'cbf' in v.name:
                vars_restore.append(v)
        # initialize the tensorflow Session


        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(var_list=vars_restore)
        saver.restore(self.sess, model_path)

    def initialize_controller(self, env):
        pass

    def reset_state(self, initial_state, opp_state):
        self.initial_state = np.expand_dims(initial_state, axis=0)  # Shape: (1, 4)
        self.opp_state = np.expand_dims(opp_state, axis=0)  # Shape: (1, 4)
    
    def make_step(self, timestamp, initial_state):
        self.initial_state = np.expand_dims(initial_state, axis=0)  # Shape: (1, 4)
        s_np = np.concatenate((self.initial_state, self.opp_state, self.static_obs_np), axis=0)
        g_np = np.concatenate((self.goal_np, np.zeros((1, 2)), self.static_obs_goals), axis=0)
        a_network, acc_list_np = self.sess.run([self.a, self.acc_list], feed_dict={self.s: s_np, self.g: g_np})    
        return np.reshape(a_network[0, :], (2, 1))
    

        

