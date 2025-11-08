#pragma once
#include <Core/Field.h>
#include <mutex>
#include <shared_mutex>

/// TODO : Field is "heavy", just use atomic<size_t> for Int32/Int64/UInt32/UInt64/Date/DateTime/Float/etc. Anything better than Field + shared_mutex.
namespace DB
{

struct TopNThresholdTracker
{
    void set(const Field & value)
    {
        std::unique_lock lock(mutex);
        threshold = value;
        is_set = true;
    }

    Field get() const
    {
        std::shared_lock lock(mutex);
        return threshold;
    }

    bool isSet() const { return is_set; } /// unlocked read is fine

private:
    Field threshold;
    mutable std::shared_mutex mutex;
    bool is_set{false};
};

using TopNThresholdTrackerPtr = std::shared_ptr<TopNThresholdTracker>;

}
