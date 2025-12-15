/**
 * @file ParticleType.h
 * @author Luis Gall
 * @date 10.12.2025
 */

#pragma once

class ParticleType {

public:
    ParticleType() {};

    enum AttributeNames : size_t {
        ptr,
        id,
        posX,
        posY,
        posZ,
        rebuildX,
        rebuildY,
        rebuildZ,
        velocityX,
        velocityY,
        velocityZ,
        forceX,
        forceY,
        forceZ,
        oldForceX,
        oldForceY,
        oldForceZ,
        typeId,
        ownershipState
      };

    template <class MemSpace>
    using KokkosSoAArraysType = autopas::utils::KokkosSoA<MemSpace, size_t* /*id*/, double* /*x*/, double* /*y*/, double* /*z*/,
                                       double* /*rebuildX*/, double* /*rebuildY*/, double* /*rebuildZ*/,
                                       double* /*vx*/, double* /*vy*/, double* /*vz*/, double* /*fx*/, double* /*fy*/,
                                       double* /*fz*/, double* /*oldFx*/, double* /*oldFy*/, double* /*oldFz*/,
                                       size_t* /*typeid*/, autopas::OwnershipState* /*ownershipState*/>;

    using SoAArraysType =
      autopas::utils::SoAType<ParticleType *, size_t /*id*/, double /*x*/, double /*y*/, double /*z*/,
                                       double /*rebuildX*/, double /*rebuildY*/, double /*rebuildZ*/,
                                       double /*vx*/, double /*vy*/, double /*vz*/, double /*fx*/, double /*fy*/,
                                       double /*fz*/, double /*oldFx*/, double /*oldFy*/, double /*oldFz*/,
                                       size_t /*typeid*/, autopas::OwnershipState /*ownershipState*/>::Type;

    template <AttributeNames attribute>
    constexpr auto& operator() () {
        auto value = get<attribute>();
        return value;
    }

    template <AttributeNames attribute, std::enable_if_t<attribute == ptr, bool> = true>
    constexpr std::tuple_element<attribute, SoAArraysType>::type::value_type get() {
        return this;
    }

    template <AttributeNames attribute, std::enable_if_t<attribute != ptr, bool> = true>
    constexpr std::tuple_element<attribute, SoAArraysType>::type::value_type get() const {
        if constexpr (attribute == id) {
            return _id;
        } else if constexpr (attribute == posX) {
            return _r[0];
        } else if constexpr (attribute == posY) {
            return _r[1];
        } else if constexpr (attribute == posZ) {
            return _r[2];
        } else if constexpr (attribute == rebuildX) {
            return _rRebuild[0];
        } else if constexpr (attribute == rebuildY) {
            return _rRebuild[1];
        } else if constexpr (attribute == rebuildZ) {
            return _rRebuild[2];
        } else if constexpr (attribute == velocityX) {
            return _v[0];
        } else if constexpr (attribute == velocityY) {
            return _v[1];
        } else if constexpr (attribute == velocityZ) {
            return _v[2];
        } else if constexpr (attribute == forceX) {
            return _f[0];
        } else if constexpr (attribute == forceY) {
            return _f[1];
        } else if constexpr (attribute == forceZ) {
            return _f[2];
        } else if constexpr (attribute == oldForceX) {
            return _oldF[0];
        } else if constexpr (attribute == oldForceY) {
            return _oldF[1];
        } else if constexpr (attribute == oldForceZ) {
            return _oldF[2];
        } else if constexpr (attribute == typeId) {
            return _typeId;
        } else if constexpr (attribute == ownershipState) {
            return _state;
        } else {
            autopas::utils::ExceptionHandler::exception("ParticleType::get() unknown attribute {}", attribute);
        }
    }

    template <AttributeNames attribute>
    constexpr void set(std::tuple_element<attribute, SoAArraysType>::type::value_type value) {
        if constexpr (attribute == id) {
            _id = value;
        } else if constexpr (attribute == posX) {
            _r[0] = value;
        } else if constexpr (attribute == posY) {
            _r[1] = value;
        } else if constexpr (attribute == posZ) {
            _r[2] = value;
        } else if constexpr (attribute == rebuildX) {
            _rRebuild[0] = value;
        } else if constexpr (attribute == rebuildY) {
            _rRebuild[1] = value;
        } else if constexpr (attribute == rebuildZ) {
            _rRebuild[2] = value;
        } else if constexpr (attribute == velocityX) {
            _v[0] = value;
        } else if constexpr (attribute == velocityY) {
            _v[1] = value;
        } else if constexpr (attribute == velocityZ) {
            _v[2] = value;
        } else if constexpr (attribute == forceX) {
            _f[0] = value;
        } else if constexpr (attribute == forceY) {
            _f[1] = value;
        } else if constexpr (attribute == forceZ) {
            _f[2] = value;
        } else if constexpr (attribute == oldForceX) {
            _oldF[0] = value;
        } else if constexpr (attribute == oldForceY) {
            _oldF[1] = value;
        } else if constexpr (attribute == oldForceZ) {
            _oldF[2] = value;
        } else if constexpr (attribute == typeId) {
            _typeId = value;
        } else if constexpr (attribute == ownershipState) {
           _state = value;
        } else {
            autopas::utils::ExceptionHandler::exception("ParticleType::set() unknown attribute {}", attribute);
        }
    }

    /* AutoPas general required params */

    const std::array<double, 3>& getR() const {
        return _r;
    }

    void setR(const std::array<double, 3>& r) {
        _r = r;
    }

    const std::array<double, 3>& getV() const {
        return _v;
    }

    void setV(const std::array<double, 3>& v) {
        _v = v;
    }

    const std::array<double, 3>& getF() const {
        return _f;
    }

    void setF(const std::array<double, 3>& f) {
        _f = f;
    }

    size_t getID() const {
        return _id;
    }

    void setID(const size_t id) {
        _id = id;
    }

    autopas::OwnershipState getOwnershipState() const {
        return _state;
    }

    void setOwnershipState(autopas::OwnershipState newState) {
        _state = newState;
    }

    std::array<double, 3> calculateDisplacementSinceRebuild() const {
        return {_rRebuild[0] - _r[0], _rRebuild[1] - _r[1], _rRebuild[2] - _r[2]};
    }

    void resetRAtRebuild() {
        _rRebuild = _r;
    }

    std::string toString() const {
        return "TODO";
    }

    bool isDummy() const {
        // TODO
        return false;
    }

    bool isHalo() const {
        // TODO
        return false;
    }

    bool isOwned() const {
        // TODO
        return false;
    }

    void markAsDeleted() {
        // TODO
    }

private:

    // TODO: replace with sth else when allowing AoS on the GPU
    std::array<double, 3> _r {};

    std::array<double, 3> _rRebuild {};

    std::array<double, 3> _v {};

    std::array<double, 3> _f {};

    std::array<double, 3> _oldF {};

    size_t _id = 0;

    size_t _typeId = 0;

    autopas::OwnershipState _state {};

};