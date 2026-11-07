// ============================================================
// <ProfilePage /> — user profile / settings
// ============================================================

import { Card, Button, Input } from "@/components/common";
import { SeekerMemory } from "@/components/profile/SeekerMemory";

export const ProfilePage = () => {
  return (
    <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
      <Card className="text-center" title="Profile" icon="👤">
        <div className="mx-auto mb-3 flex h-24 w-24 items-center justify-center rounded-full bg-gradient-to-br from-amber-500 to-amber-700 text-4xl text-amber-950 shadow-lg">
          𓁹
        </div>
        <h3 className="text-lg font-bold text-amber-200">Seeker of Wisdom</h3>
        <p className="text-xs text-amber-400/80">scholar plan · joined 2026</p>
        <div className="mt-4">
          <Button size="sm" variant="secondary">
            Edit avatar
          </Button>
        </div>
      </Card>

      <Card className="lg:col-span-2" title="Account" icon="⚙">
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
          <label className="block text-sm">
            <span className="mb-1 block text-amber-300/80">Display name</span>
            <Input defaultValue="Seeker of Wisdom" />
          </label>
          <label className="block text-sm">
            <span className="mb-1 block text-amber-300/80">Email</span>
            <Input type="email" defaultValue="seeker@sphinxeyes.ai" />
          </label>
          <label className="block text-sm sm:col-span-2">
            <span className="mb-1 block text-amber-300/80">Research interests</span>
            <Input defaultValue="Pyramids, hieroglyphs, mythology" />
          </label>
        </div>
        <div className="mt-5 flex justify-end gap-2">
          <Button variant="secondary">Cancel</Button>
          <Button>Save changes</Button>
        </div>
      </Card>

      <SeekerMemory />
    </div>
  );
};

export default ProfilePage;
