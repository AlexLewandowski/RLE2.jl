import GridWorlds
import GridWorlds.GridRoomsUndirectedModule: GridRoomsUndirected, AGENT, GOAL
import GridWorlds.GridRoomsDirectedModule: GridRoomsDirected, AGENT, GOAL
import GridWorlds: RLBaseEnv

abstract type AbstractGridWorld <: AbstractMDP end
abstract type AbstractGridWorldUndirected <: AbstractGridWorld end
abstract type AbstractGridWorldDirected <: AbstractGridWorld end

mutable struct GridWorldDirected{E} <: AbstractGridWorldDirected
    env::RLBaseEnv{E}
end

mutable struct GridWorldUndirected{E} <: AbstractGridWorldUndirected
    env::RLBaseEnv{E}
end

mutable struct TabularGridWorld{E} <: AbstractGridWorld
    env::E
    state_transition_mat
    reward_mat
    action
    reward
    t
    V
    gamma
    valid_states
    name
    info
    rng::AbstractRNG
end

GridRooms = Union{GridRoomsUndirected, GridRoomsDirected}

Base.show(io::IO,t::MIME"text/markdown", env::TabularGridWorld) = begin
    Base.show(io, t, env.env)
end

function TabularGridWorld(EnvType; rng = Random.GLOBAL_RNG, directed = false, negative_reward = true, kwargs...)
    info = Dict()
    info["p"] = 0.7f0
    if EnvType == "FourRooms"
        goal = (8,8)
        if directed
            env = GridRoomsDirected(rng = rng)
        else
            env = GridRoomsUndirected(rng = rng)
        end
    elseif EnvType == "SingleRoom"
        goal = (7,7)
        if directed
            env = GridWorlds.SingleRoomDirectedModule.SingleRoomDirected(rng = rng)
        else
            env = GridWorlds.SingleRoomUndirectedModule.SingleRoomUndirected(rng = rng)
        end
    elseif EnvType == "CompassWorld"
        goal = (7,7)
        directed = true
        env = GridWorlds.SingleRoomDirectedModule.SingleRoomDirected(rng = rng)
    else
        error("Not a valid GridWorld.")
    end
    if directed
        env = GridWorldDirected(RLBaseEnv(env))
    else
        env = GridWorldUndirected(RLBaseEnv(env))
    end

    E = typeof(env)

    world = get_tilemap(env)
    walls = world[2,:,:]
    X,Y = size(walls)
    wall_flat = Int.(reshape(walls, :))
    valid_states = get_valid_states(env)
    n_states = get_states(env)
    n_orientations = directed ? 4 : 1
    n_tiles = X*Y
    n_actions = GridWorlds.RLBase.action_space(env.env).stop

    state_transition_mat = reshape(
        [zeros(Float32, n_states) for i = 1:n_states*n_actions],
        (n_states, n_actions),
    )

    reward_mat =
        Float32.(
            reshape(
                -zeros(Float32, (n_states) * n_actions),
                (n_states, n_actions),
            ),
        )/n_actions

    temp = zeros(Float32, n_states)
    tiled_temp = reshape(temp, (X,Y,n_orientations))
    tiled_temp[goal...,:] .= 1f0
    goal_flat = findall(reshape(tiled_temp, :) .== 1f0)

    for s = 1:n_states
        temp = zeros(Float32, n_states)
        temp[s] = 1f0
        tiled_temp = reshape(temp, (X,Y,n_orientations))
        local_y, local_x, orientation = findall(tiled_temp .== 1f0)[1].I
        for a = 1:n_actions
            xycord = get_next_state(env, a, local_y, local_x, orientation)

            temp = zeros(Float32, n_states)

            if s in goal_flat
                temp = reshape(zeros(Float32, n_states), (X,Y, n_orientations))
                temp[local_y, local_x, orientation] = 1
                temp = reshape(temp, :)
                state_transition_mat[s,a] = copy(temp)
            elseif any(xycord[1:2] .== 0)
                temp = reshape(zeros(Float32, n_states), (X,Y, n_orientations))
                temp[local_y, local_x, orientation] = 1
                temp = reshape(temp, :)
                state_transition_mat[s,a] = copy(temp)

            elseif any(xycord[1:2] .== X + 1)
                temp = reshape(zeros(Float32, n_states), (X,Y, n_orientations))
                temp[local_y, local_x, orientation] = 1
                temp = reshape(temp, :)
                state_transition_mat[s,a] = copy(temp)
            else
                ind_xy = xy2int(env, xycord)
                temp = reshape(zeros(Float32, n_states), (X,Y, n_orientations))
                if local_x == 8 && local_y == 8 && !negative_reward && s in get_valid_nonterminal_states(env)
                    reward_mat[s, a] = 1f0
                end
                if local_y == 8 && local_x == 8
                    temp[local_y, local_x, orientation] = 1
                    temp = reshape(temp, :)
                    state_transition_mat[s,a] = copy(temp)
                elseif ind_xy == goal_flat
                    temp[xycord...] = 1
                    temp = reshape(temp, :)
                    state_transition_mat[s,a] = copy(temp)
                    reward_mat[s,a] = 1f0
                elseif ind_xy in get_valid_nonterminal_states(env)
                    temp[xycord...] = 1
                    temp = reshape(temp, :)
                    state_transition_mat[s,a] = copy(temp)
                else
                    temp[local_y, local_x, orientation] = 1
                    temp = reshape(temp, :)
                    state_transition_mat[s,a] = copy(temp)
                end
            end
        end
    end

    if negative_reward
        reward_mat .-= 1f0
        reward_rng = MersenneTwister(1) #TODO ? Keep rewards consistent between runs
        # reward_mat -= 2*rand(reward_rng, Float32, size(reward_mat))

        reward_mat = reshape(reward_mat, (X,Y,n_orientations,n_actions))
        reward_mat[goal...,:, :] .= 0f0
        reward_mat = reshape(reward_mat, (n_states, n_actions))
    end

    if EnvType == "CompassWorld"

        reward_mat =
            Float32.(
                reshape(
                    -zeros(Float32, (n_states) * n_actions),
                    (n_states, n_actions),
                ),
            )/n_actions

        for s = 1:n_states
            for a = 1:n_actions
                if a < 3
                    sp = argmax(state_transition_mat[s,a])
                    xyd = int2xy(env, sp)
                    X = xyd[2]
                    Y = xyd[1]
                    reward_mat[s,a] -= (X-Y)^2
                end
            end
        end

        reward_mat = 2*reward_mat ./ (abs(minimum(reward_mat)))

        reward_mat .-= 1
        reward_mat -= rand(reward_rng, Float32, size(reward_mat))

        # for i = 2:6
        #     cord = (i, i, 2)
        #     s = xy2int(env, cord)
        #     reward_mat[s, 4] = -0.01f0

        #     cord = (i, i, 1)
        #     s = xy2int(env, cord)
        #     reward_mat[s, 1] = -0.01f0

        #     cord = (i, i + 1, 1)
        #     s = xy2int(env, cord)
        #     reward_mat[s, 3] = -0.01f0

        #     cord = (i, i + 1, 2)
        #     s = xy2int(env, cord)
        #     reward_mat[s, 2] = -0.01f0
        # end

        reward_mat = reshape(reward_mat, (X,Y,n_orientations,n_actions))
        reward_mat[goal...,:, :] .= 0f0
        reward_mat = reshape(reward_mat, (n_states, n_actions))

    end

    V = zeros(Float32, n_states)
    t = 0f0
    gamma = 0.99f0
    action = 1
    reward = 0f0
    name = EnvType
    env = TabularGridWorld{E}(env, state_transition_mat, reward_mat, action, reward, t, V, gamma, valid_states, name, info, rng)
    env.action = random_action(env)
    reset!(env)
    env.V = get_optimal_V(env)
    return env
end

function env_mode!(env::AbstractMDP; mode = :default)
    test_env = deepcopy(env)
    if mode == :default || mode == :train
        test_env.info["p"] = 0.8f0
        return test_env
    else
        test_env.info["p"] = 0.0f0
        return test_env
    end
end

function get_next_state(env::AbstractGridWorldDirected, a, local_y, local_x, orientation)
    xycord = nothing
    if a == 1
        if orientation == 1
            xycord = (local_y, local_x + 1, orientation)
        elseif orientation == 2
            xycord = (local_y - 1, local_x, orientation)
        elseif orientation == 3
            xycord = (local_y, local_x - 1, orientation)
        elseif orientation == 4
            xycord = (local_y + 1, local_x, orientation)
        end

    elseif a == 2
        if orientation == 1
            xycord = (local_y, local_x - 1, orientation)
        elseif orientation == 2
            xycord = (local_y + 1, local_x, orientation)
        elseif orientation == 3
            xycord = (local_y, local_x + 1, orientation)
        elseif orientation == 4
            xycord = (local_y - 1, local_x, orientation)
        end
    elseif a == 3
        new_orientation = mod1(orientation + 1, 4)
        xycord = (local_y, local_x, new_orientation)
    elseif a == 4
        new_orientation = mod1(orientation - 1, 4)
        xycord = (local_y, local_x, new_orientation)
    else
        error()
    end
    return xycord
end

function get_next_state(env::AbstractGridWorldUndirected, a, local_y, local_x, orientation)
    xycord = nothing
    if a == 1
        xycord = (local_y - 1, local_x, orientation)
    elseif a == 2
        xycord = (local_y + 1, local_x, orientation)
    elseif a == 3
        xycord = (local_y, local_x - 1, orientation)
    elseif a == 4
        xycord = (local_y, local_x + 1, orientation)
    else
        error()
    end
    return xycord
end

function get_states(env::AbstractGridWorldDirected)
    n_actions = GridWorlds.RLBase.action_space(env.env).stop
    walls = get_tilemap(env)[2,:,:]
    X,Y = size(walls)
    n_states = n_actions*X*Y
    return n_states
end

function get_states(env::AbstractGridWorldUndirected)
    walls = get_tilemap(env)[2,:,:]
    X,Y = size(walls)
    n_states = X*Y
    return n_states
end

function int2xy(env::AbstractGridWorld, x::Int)
    n_orientations = typeof(env) <: AbstractGridWorldDirected ? 4 : 1
    walls = get_tilemap(env)[2,:,:]
    X,Y = size(walls)
    wall_flat = Int.(reshape(walls, :))
    n_states = get_states(env)
    temp = zeros(Float32, n_states)
    temp[x] = 1f0
    return argmax(reshape(temp, (X,Y,n_orientations)))
end

function int2xy(env::TabularGridWorld, x::Int)
    int2xy(env.env, x)
end

function xy2int(env::AbstractGridWorld, x::Tuple)
    n_orientations = typeof(env) <: AbstractGridWorldDirected ? 4 : 1
    walls = get_tilemap(env)[2,:,:]
    X,Y = size(walls)
    wall_flat = Int.(reshape(walls, :))
    n_states = get_states(env)
    temp = zeros(Float32, n_states)
    temp =  reshape(temp, (X,Y, n_orientations))
    temp[x...] = 1f0
    return argmax(reshape(temp, :))
end

function xy2int(env::TabularGridWorld, x::Tuple)
    xy2int(env.env, x)
end

function (env::TabularGridWorld)(a::Int)
    # if :agent_dir in fieldnames(typeof(env))
    #     if a == 1
    #         a = GridWorlds.MOVE_FORWARD()
    #     elseif a == 2
    #         a = GridWorlds.TURN_LEFT
    #     elseif a == 3
    #         a = GridWorlds.TURN_RIGHT
    #     else
    #         error("Not a valid action")
    #     end
    # else
    #     if a == 1
    #         a = GridWorlds.move_up(env.env.agent_position)
    #     elseif a == 2
    #         a = GridWorlds.move_down(env.env.agent_position)
        #     elseif a == 3
    #         a = GridWorlds.move_right(env.env.agent_position)
    #     elseif a == 4
    #         a = GridWorlds.move_left(env.env.agent_position)
    #     else
    #         error("Not a valid action")
    #     end
    # end
    # GridWorlds.act!(env.env,a)
    env.action = a
    s = get_state(env)
    env.reward = env.reward_mat[s, a]
    env.env.env(a)
end


function reset!(env::TabularGridWorld; goal_pos = :default, agent_pos = :default)
    reset!(env.env.env)
    world = get_tilemap(env.env)
    rng = get_rng(env)

    height = size(world)[2]
    width = size(world)[3]

    world[AGENT, get_agent_pos(env.env)] = false
    world[GOAL, get_goal_pos(env.env)] = false

    if goal_pos == :default
        new_goal_pos = CartesianIndex(height -1, width - 1)
    elseif goal_pos == :random
        new_goal_pos = rand(rng, pos -> !any(@view env.env.world[:, pos]), env)
    else
        new_goal_pos = goal_pos
    end

    set_goal_pos!(env.env, new_goal_pos)
    world[GOAL, new_goal_pos] = true

    if agent_pos == :default
        new_agent_pos = discrete_agent_pos(env, default_action(env))
    elseif agent_pos == :random
        new_agent_pos = rand(rng, pos -> !any(@view world[:, pos]), env.env)
    elseif typeof(agent_pos) <: Int
        new_agent_pos = discrete_agent_pos(env, agent_pos)
    elseif typeof(agent_pos) <: CartesianIndex
        new_agent_pos = agent_pos
    elseif typeof(agent_pos) <: Tuple
        new_agent_pos = agent_pos
    end

    try
        get_tilemap(env.env)[:, new_agent_pos.I...]
    catch
        println(new_agent_pos)
        error("Not a valid coordinate:")
    end
    set_agent_pos!(env.env, new_agent_pos)
    world[AGENT, new_agent_pos] = true

    return nothing
end

function Random.seed!(env::TabularGridWorld, seed)
    # env.rng = MersenneTwister(seed)
end

function optimal_action(env::AbstractGridWorld, s)
    a = get_optimal_action(env, s)[1][1]
end

function optimal_action(env::AbstractGridWorld)
    s = get_state(env)
    a = get_optimal_action(env, s)[1][1]
end

function get_state(env::TabularGridWorld{E}) where {E <: AbstractGridWorldDirected}
    map = get_tilemap(env.env)[1, :, :]
    z = zeros(Float32, (size(map)..., 1))
    dir = env.env.env.env.agent_direction
    obs = [z for i = 1:4]
    dir = env.env.env.env.agent_direction + 1
    obs[dir] = reshape(map, (size(map)..., 1))
    return argmax(reshape(cat(obs..., dims = 3), :))
    # return ReinforcementLearningEnvironments.state(env)
end

function get_state(env::TabularGridWorld{E}) where {E <: AbstractGridWorldUndirected}
    map = get_tilemap(env.env)[1, :, :]
    return argmax(reshape(map, :))
end

function get_obs(env::TabularGridWorld{E}) where {E<:AbstractGridWorldUndirected}
    s = zeros(Bool, size(env.reward_mat)[1])
    state = get_state(env)
    s[state] = 1
    return s
end

function get_obs(env::TabularGridWorld{E}) where {E<:AbstractGridWorldDirected}
    X,Y = size(get_tilemap(env.env)[1, :, :])
    if env.name == "CompassWorld"
        s = get_state(env)
        xycord = int2xy(env, s)
        dir = env.env.env.env.agent_direction + 1
        noise = [0f0, 0f0, 0f0, 0f0]
        if dir == 1
            cord = xycord[2]
            dist = X - cord
            if dist > 1
                optimal_a = 1
            else
                optimal_a = 4
            end
        elseif dir == 2
            cord = xycord[1]
            dist = cord - 1
            optimal_a = 4
        elseif dir == 3
            cord = xycord[2]
            dist = cord - 1
            optimal_a = 3
        elseif dir == 4
            cord = xycord[1]
            dist = Y - cord
            if dist > 1
                optimal_a = 1
            else
                optimal_a = 3
            end
        end

        noise2 = [0f0, 0f0, 0f0, 0f0]
        if dir == 1
            optimal_a2 = 3
        elseif dir == 2
            cord = xycord[1]
            dist = cord - 1
            if dist < 6
                optimal_a2 = 2
            else
                optimal_a2 = 3
            end
        elseif dir == 3
            cord = xycord[2]
            dist = cord - 1
            if dist < 6
                optimal_a2 = 2
            else
                optimal_a2 = 4
            end
        elseif dir == 4
            optimal_a2 = 4
        end

        if rand() < env.info["p"]
            noise[optimal_a] = 1f0
        end

        if rand() < env.info["p"]
            noise2[optimal_a2] = 1f0
        end

        Ycord = xycord[1]
        Xcord = xycord[2]

        Yvec = Float32.(discrete_action_mask(Ycord, Y))
        Xvec = Float32.(discrete_action_mask(Xcord, X))

        z = zeros(Float32, 4)

        z[dir] = 1.0f0

        # z = vcat(z, Xvec)
        # z = vcat(z, Yvec)

        z = vcat(z, noise)
        z = vcat(z, noise2)

        z = vcat(z, Xcord)
        z = vcat(z, Ycord)

        return z
    else
        map = get_tilemap(env.env)[1, :, :]
        z = zeros(Float32, (size(map)..., 1))
        dir = env.env.env.env.agent_direction
        obs = [z for _ = 1:4]
        dir = env.env.env.env.agent_direction + 1
        obs[dir] = reshape(map, (size(map)..., 1))
        return reshape(cat(obs..., dims = 3), :)
    end
    # return ReinforcementLearningEnvironments.state(env)
end

function get_terminal(env::TabularGridWorld)
    return GridWorlds.RLBase.is_terminated(env.env.env)
end

function get_reward(env::TabularGridWorld)
    return env.reward[1]
end

function get_actions(env::TabularGridWorld{E}) where {E<:AbstractGridWorldUndirected}
    return Base.OneTo(4)
end

function get_actions(env::TabularGridWorld{E}) where {E <: AbstractGridWorldDirected}
    return Base.OneTo(4)
end

function random_action(env::TabularGridWorld)
    return rand(env.rng, get_actions(env))
end

function get_tilemap(env::TabularGridWorld)
    get_tilemap(env.env)
end

function get_tilemap(env::GridWorldUndirected)
    env.env.env.tile_map
end

function get_tilemap(env::GridWorldDirected)
    env.env.env.env.tile_map
end

function get_rng(env::TabularGridWorld)
    env.rng
end

function get_agent_pos(env::GridWorldUndirected)
    env.env.env.agent_position
end

function get_agent_pos(env::GridWorldDirected)
    env.env.env.env.agent_position
end

function get_goal_pos(env::GridWorldUndirected)
    env.env.env.goal_position
end

function get_goal_pos(env::GridWorldDirected)
    env.env.env.env.goal_position
end

function set_goal_pos!(env::GridWorldUndirected, goal)
    env.env.env.goal_position = goal
end

function set_goal_pos!(env::GridWorldDirected, goal)
    env.env.env.env.goal_position = goal
end

function set_agent_pos!(env::GridWorldUndirected, agent_pos)
    env.env.env.agent_position = CartesianIndex(agent_pos.I[1:2])
end

function set_agent_pos!(env::GridWorldDirected, agent_pos)
    env.env.env.env.agent_position = CartesianIndex(agent_pos.I[1:2])
    env.env.env.agent_direction = agent_pos.I[3]
    # env.env.env.agent_direction = rand(env.env.env.env.rng, 0:3) #TODO deterministic init of direction?
end

function get_all_states(env::TabularGridWorld)
    R = env.reward_mat
    n_states = size(R)[1]
    return collect(1:n_states)
end

function get_valid_nonterminal_states(env::AbstractGridWorldUndirected)
    wall = get_tilemap(env)[2,:,:]
    wall[8,8] = 1 # Since goal can sometimes be somewhere else by default
    wall = reshape(wall, :)
    inds = findall(wall .== 0)
    return inds
end

function get_valid_states(env::AbstractGridWorldUndirected)
    wall = get_tilemap(env)[2,:,:]
    wall = reshape(wall, :)
    inds = findall(wall .== 0)
    return inds
end

function get_valid_nonterminal_states(env::AbstractGridWorldDirected)
    wall = get_tilemap(env)[2,:,:]
    wall[8,8] = 1 # Since goal can sometimes be somewhere else by default
    wall = cat([reshape(wall, (size(wall)...,1)) for i = 1:4]..., dims = 3)
    wall = reshape(wall, :)
    inds = findall(wall .== 0)
    return inds
end

function get_valid_states(env::AbstractGridWorldDirected)
    wall = get_tilemap(env)[2,:,:]
    wall = cat([reshape(wall, (size(wall)...,1)) for i = 1:4]..., dims = 3)
    wall = reshape(wall, :)
    inds = findall(wall .== 0)
    return inds
end

function get_valid_states(env::TabularGridWorld)
    get_valid_states(env.env)
end

function get_valid_nonterminal_states(env::TabularGridWorld)
    get_valid_nonterminal_states(env.env)
end

function get_wall_states(env::AbstractGridWorld)
    wall = get_tilemap(env)[2,:,:]
    inds = findall(wall .== 1)
    return [xy2int(env, ind.I) for ind in inds]
end

function get_wall_states(env::TabularGridWorld)
    get_wall_states(env.env)
end

function get_valid_wall_states(env::AbstractGridWorld)
    wall = get_tilemap(env)[2,:,:]
    X,Y = size(wall)
    inds = findall(wall .== 1)
    inds = [ind for ind in inds if !(1 in ind.I || X in ind.I || Y in ind.I ) &&
        (!wall[ind.I[1] + 1, ind.I[2]]     ||
         !wall[ind.I[1] - 1, ind.I[2]]     ||
         !wall[ind.I[1]    , ind.I[2] + 1] ||
         !wall[ind.I[1]    , ind.I[2] - 1] )]
    return [xy2int(env, ind.I) for ind in inds]
end

function get_valid_wall_states(env::TabularGridWorld)
    return get_valid_wall_states(env.env)
end

function get_goal_states(env::GridWorldUndirected)
    return [env.env.env.goal_position.I]
end

function get_goal_states(env::GridWorldDirected)
    G =  env.env.env.env.goal_position.I
    return [tuple(vcat(G..., i)...) for i = 1:4]
end

function get_goal_states(env::TabularGridWorld)
    ss = get_goal_states(env.env)
    states =  [xy2int(env.env, s) for s in ss]
    valid_states = get_valid_states(env)
    return [argmax(valid_states .== state) for state in states]
end

function compass_clockwise_expert(env::TabularGridWorld)
    s = get_state(env)
    xy = int2xy(env, s)
    if s == 10
        a = 4
    end
    if 2 <= xy[1] < 7 && xy[2] == 2 && xy[3] == 4
        a = 1
    end

    if xy[1] == 7 && xy[2] == 2 && xy[3] !== 1
        a = 4
    end

    if xy[1] == 7 && 2 <= xy[2] < 7 && xy[3] == 1
        a = 1
    end
    return a, 1f0
end

function compass_counterclockwise_expert(env::TabularGridWorld)
    s = get_state(env)
    xy = int2xy(env, s)
    if xy[1] == 2 && xy[2] == 2 && xy[3] !== 4
        a = 3
    end

    xy = int2xy(env, s)
    if 2 <= xy[1] < 7 && xy[2] == 2 && xy[3] == 4
        a = 1
    end

    if xy[1] == 7 && xy[2] == 2 && xy[3] !== 1
        a = 3
    end

    if xy[1] == 7 && 2 <= xy[2] < 7 && xy[3] == 1
        a = 1
    end
    return a, 1f0
end

function compass_policy_1_rng(env::Union{TabularGridWorld, AbstractContextualMDP{E}}) where {E<:TabularGridWorld}
    as = [1,2,3,4]
    o = get_obs(env)

    noise = o[5:8]

    rng = env.rng

    if sum(noise) == 1
        return argmax(noise), 1f0
    else
        return rand(rng, as), 0.25f0
    end
end

function compass_policy_1_det(env::Union{TabularGridWorld, AbstractContextualMDP{E}}) where {E<:TabularGridWorld}
    as1 = [1,2]
    as2 = [3,4]
    o = get_obs(env)

    noise = o[5:8]

    s = get_state(env)
    rng = MersenneTwister(s)
    seed = rand(rng, 1:1000000)
    rng = MersenneTwister(seed + 1)

    if sum(noise) == 1
        return argmax(noise), 1f0
    else
        not_a = rand(rng, [1,2])
        deleteat!(as1, not_a)
        not_a = rand(rng, [1,2])
        deleteat!(as2, not_a)
        return rand(env.rng, [as1[1], as2[1]]), 0.25f0
    end
end

function compass_policy_2_rng(env::Union{TabularGridWorld, AbstractContextualMDP{E}}) where {E<:TabularGridWorld}
    as = [1,2,3,4]
    o = get_obs(env)

    noise = o[9:12]

    rng = env.rng

    if sum(noise) == 1
        return argmax(noise), 1f0
    else
        return rand(rng, as), 0.25f0
    end
end

function compass_policy_2_det(env::Union{TabularGridWorld, AbstractContextualMDP{E}}) where {E<:TabularGridWorld}
    as1 = [1,2]
    as2 = [3,4]
    o = get_obs(env)

    noise = o[9:12]

    s = get_state(env)
    rng = MersenneTwister(s)
    seed = rand(rng, 1:1000000)
    rng = MersenneTwister(10*seed - 1)

    if sum(noise) == 1
        return argmax(noise), 1f0
    else
        not_a = rand(rng, [1,2])
        deleteat!(as1, not_a)
        not_a = rand(rng, [1,2])
        deleteat!(as2, not_a)
        return rand(env.rng, [as1[1], as2[1]]), 0.25f0
    end
end

function compass_policy_2_det_same(env::Union{TabularGridWorld, AbstractContextualMDP{E}}) where {E<:TabularGridWorld}
    as1 = [1,2]
    as2 = [3,4]
    o = get_obs(env)

    noise = o[9:12]

    s = get_state(env)
    rng = MersenneTwister(s)
    seed = rand(rng, 1:1000000)
    rng = MersenneTwister(seed + 1)

    if sum(noise) == 1
        return argmax(noise), 1f0
    else
        not_a = rand(rng, [1,2])
        deleteat!(as1, not_a)
        not_a = rand(rng, [1,2])
        deleteat!(as2, not_a)
        return rand(env.rng, [as1[1], as2[1]]), 0.25f0
    end
end

function compass_policy_2_det_diff(env::Union{TabularGridWorld, AbstractContextualMDP{E}}) where {E<:TabularGridWorld}
    as1 = [1,2]
    as2 = [3,4]
    o = get_obs(env)

    noise = o[9:12]

    s = get_state(env)
    rng = MersenneTwister(s)
    seed = rand(rng, 1:1000000)
    rng = MersenneTwister(seed + 1)

    if sum(noise) == 1
        return argmax(noise), 1f0
    else
        not_a = rand(rng, [2,1])
        deleteat!(as1, not_a)
        not_a = rand(rng, [2,1])
        deleteat!(as2, not_a)
        return rand(env.rng, [as1[1], as2[1]]), 0.25f0
    end
end
