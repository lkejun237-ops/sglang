(function (global) {
  const FEISHU_INTERNAL_DEMO_ASSET_BASE_URL = "./assets/presets/v1/feishu-internal-demo-20260825";
  global.FEISHU_INTERNAL_DEMO_PRESETS = [
  {
    "name": "Phantom Wings over Pandora",
    "tone": "green",
    "size": "832x480",
    "fps": 24,
    "prompt": "A third-person trailing gameplay view keeps the clear external playable subject centered in the frame on the central vertical axis, seen from behind and facing the route ahead. This third-person trailing gameplay view positions the camera four meters behind the playable subject, who is centered in the frame on the central vertical axis and occupies approximately 25% of the frame height. The subject is a slender Na'vi avatar with blue-toned skin, wearing minimal organic tribal gear and leather straps, mounted atop a large, iridescent blue mountain banshee with expansive, translucent wings that shimmer with bioluminescent patterns. The rider faces directly away from the camera, guiding the creature toward a vast, traversable canyon floor below. The environment features a continuous, winding river path that serves as a primary route through the lush, dense jungle floor, leading toward distant, mist-covered mountain passes. The terrain is characterized by towering, moss-covered floating rock formations that create a vertical, multi-layered landscape. Numerous waterfalls cascade from the edges of these floating islands into the valley, creating a constant, ethereal mist that catches the soft, ambient light. The atmosphere is saturated with glowing, bioluminescent particles and a soft, cyan-tinted light emanating from a massive, eye-like celestial anomaly in the sky. The color palette is dominated by deep teals, vibrant blues, and lush forest greens, creating a surreal, high-fantasy aesthetic. The lighting is diffuse and atmospheric, casting long, soft shadows across the dense canopy and the winding river below, emphasizing the immense scale of the floating cliffs and the depth of the valley. The scene is peaceful yet grand, inviting exploration of the hidden paths beneath the floating mountains.",
    "referenceUrl": `${FEISHU_INTERNAL_DEMO_ASSET_BASE_URL}/images/row-002-pandora-banshee.jpg`,
    "mime": "image/jpeg",
    "source": "Feishu 抽卡showcase row 2",
    "worldDefinitionUrl": `${FEISHU_INTERNAL_DEMO_ASSET_BASE_URL}/worlds/row-002-pandora-banshee.json`,
    "worldDefinition": {
      "schemaVersion": "loopit.world/v1",
      "mode": "experience",
      "id": "pandora-banshee",
      "version": "0.1.0",
      "creationInput": "Let me ride a banshee through floating canyons and waterfalls beneath Pandora's giant sky eye.",
      "presentation": {
        "title": "Phantom Wings over Pandora",
        "introduction": "The old flight paths have vanished among Pandora’s floating peaks. Mount a banshee and follow the waterfalls into the wilds beyond."
      },
      "world": {
        "initialSetup": {
          "perspective": "third_person",
          "startingFrame": {
            "source": "upload",
            "uri": "asset://pending"
          },
          "description": "A third-person trailing gameplay view keeps the clear external playable subject centered in the frame on the central vertical axis, seen from behind and facing the route ahead. This third-person trailing gameplay view positions the camera four meters behind the playable subject, who is centered in the frame on the central vertical axis and occupies approximately 25% of the frame height. The subject is a slender Na'vi avatar with blue-toned skin, wearing minimal organic tribal gear and leather straps, mounted atop a large, iridescent blue mountain banshee with expansive, translucent wings that shimmer with bioluminescent patterns. The rider faces directly away from the camera, guiding the creature toward a vast, traversable canyon floor below. The environment features a continuous, winding river path that serves as a primary route through the lush, dense jungle floor, leading toward distant, mist-covered mountain passes. The terrain is characterized by towering, moss-covered floating rock formations that create a vertical, multi-layered landscape. Numerous waterfalls cascade from the edges of these floating islands into the valley, creating a constant, ethereal mist that catches the soft, ambient light. The atmosphere is saturated with glowing, bioluminescent particles and a soft, cyan-tinted light emanating from a massive, eye-like celestial anomaly in the sky. The color palette is dominated by deep teals, vibrant blues, and lush forest greens, creating a surreal, high-fantasy aesthetic. The lighting is diffuse and atmospheric, casting long, soft shadows across the dense canopy and the winding river below, emphasizing the immense scale of the floating cliffs and the depth of the valley. The scene is peaceful yet grand, inviting exploration of the hidden paths beneath the floating mountains."
        }
      },
      "execution": {
        "durationSeconds": 60,
        "allowFreePrompt": false,
        "maxSnapshots": 3
      },
      "skills": [
        {
          "id": "fire_breath",
          "name": "🔥 Fire Breath",
          "prompt": "The banshee exhales a long jet of vivid orange fire into the open air ahead; the flame lights the mist and fades without burning the rider or blocking the route.",
          "changeType": "one_time"
        },
        {
          "id": "wing_glow",
          "name": "✨ Wing Glow",
          "prompt": "Bioluminescent patterns brighten across the banshee's wings and body, casting vivid cyan light through the surrounding mist.",
          "changeType": "persistent"
        },
        {
          "id": "waterfall_dive",
          "name": "🌊 Waterfall Dive",
          "prompt": "The banshee dives beside a tall waterfall, skims through the spray, then climbs smoothly back into the open flight path.",
          "changeType": "one_time"
        }
      ]
    },
    "rules": {
      "skills": [
        {
          "id": "fire-breath",
          "input": "🔥 Fire Breath",
          "name": "🔥 Fire Breath",
          "prepared": {
            "prompt": "The banshee exhales a long jet of vivid orange fire into the open air ahead; the flame lights the mist and fades without burning the rider or blocking the route.",
            "change_type": "one_time"
          }
        },
        {
          "id": "wing-glow",
          "input": "✨ Wing Glow",
          "name": "✨ Wing Glow",
          "prepared": {
            "prompt": "Bioluminescent patterns brighten across the banshee's wings and body, casting vivid cyan light through the surrounding mist.",
            "change_type": "persistent"
          }
        },
        {
          "id": "waterfall-dive",
          "input": "🌊 Waterfall Dive",
          "name": "🌊 Waterfall Dive",
          "prepared": {
            "prompt": "The banshee dives beside a tall waterfall, skims through the spray, then climbs smoothly back into the open flight path.",
            "change_type": "one_time"
          }
        }
      ]
    }
  },
  {
    "name": "The Last Broadcast of Channel Zero",
    "tone": "accent",
    "size": "832x480",
    "fps": 24,
    "prompt": "A third-person trailing gameplay view keeps the clear external playable subject centered in the frame on the central vertical axis, seen from behind and facing the route ahead. The camera uses a third-person perspective, positioned low to the ground and trailing behind the subject to follow its forward movement through a vast, enclosed space. The playable subject, a small brown mouse, is centered in the frame on the central vertical axis, occupying approximately 25% of the frame height. The mouse is depicted from behind, showing its soft, textured fur, large rounded ears, and a long, slender tail that rests on the ground. It wears a small, worn yellow backpack with visible straps and buckles. The mouse stands on a dark, slightly damp floor that serves as a continuous, traversable path leading directly into the depths of a massive, infinite tunnel. The tunnel walls, ceiling, and floor are constructed from a dense, chaotic arrangement of stacked, vintage CRT television sets. These screens display a variety of original, abstract animations including rotating geometric planets, drifting clouds, blinking eyes, and shimmering starfields. The screens activate in a sequential wave from the foreground toward the vanishing point, casting vibrant, shifting pulses of cyan, magenta, and green light across the mouse's fur, the yellow backpack, and the reflective puddles on the ground. Thick, dark electrical cables snake across the floor like vines, adding to the industrial, surreal atmosphere. Dust motes dance in the flickering light beams, and the air is filled with a sense of mysterious, quiet wonder. The lighting is entirely provided by the glowing screens, creating deep shadows and high-contrast reflections on the glass surfaces of the monitors. The visual style is cinematic and highly detailed, emphasizing the contrast between the small, organic subject and the overwhelming, technological scale of the tunnel, where the stacked television sets act as solid, impassable walls.",
    "referenceUrl": `${FEISHU_INTERNAL_DEMO_ASSET_BASE_URL}/images/row-003-channel-zero-tunnel.jpg`,
    "mime": "image/jpeg",
    "source": "Feishu 抽卡showcase row 3",
    "worldDefinitionUrl": `${FEISHU_INTERNAL_DEMO_ASSET_BASE_URL}/worlds/row-003-channel-zero-tunnel.json`,
    "worldDefinition": {
      "schemaVersion": "loopit.world/v1",
      "mode": "experience",
      "id": "channel-zero-tunnel",
      "version": "0.1.0",
      "creationInput": "Let me explore an endless tunnel built from vintage televisions as a tiny mouse with a yellow backpack.",
      "presentation": {
        "title": "The Last Broadcast of Channel Zero",
        "introduction": "Channel Zero is still transmitting from somewhere inside the endless television tunnel. A tiny traveler follows the signal."
      },
      "world": {
        "initialSetup": {
          "perspective": "third_person",
          "startingFrame": {
            "source": "upload",
            "uri": "asset://pending"
          },
          "description": "A third-person trailing gameplay view keeps the clear external playable subject centered in the frame on the central vertical axis, seen from behind and facing the route ahead. The camera uses a third-person perspective, positioned low to the ground and trailing behind the subject to follow its forward movement through a vast, enclosed space. The playable subject, a small brown mouse, is centered in the frame on the central vertical axis, occupying approximately 25% of the frame height. The mouse is depicted from behind, showing its soft, textured fur, large rounded ears, and a long, slender tail that rests on the ground. It wears a small, worn yellow backpack with visible straps and buckles. The mouse stands on a dark, slightly damp floor that serves as a continuous, traversable path leading directly into the depths of a massive, infinite tunnel. The tunnel walls, ceiling, and floor are constructed from a dense, chaotic arrangement of stacked, vintage CRT television sets. These screens display a variety of original, abstract animations including rotating geometric planets, drifting clouds, blinking eyes, and shimmering starfields. The screens activate in a sequential wave from the foreground toward the vanishing point, casting vibrant, shifting pulses of cyan, magenta, and green light across the mouse's fur, the yellow backpack, and the reflective puddles on the ground. Thick, dark electrical cables snake across the floor like vines, adding to the industrial, surreal atmosphere. Dust motes dance in the flickering light beams, and the air is filled with a sense of mysterious, quiet wonder. The lighting is entirely provided by the glowing screens, creating deep shadows and high-contrast reflections on the glass surfaces of the monitors. The visual style is cinematic and highly detailed, emphasizing the contrast between the small, organic subject and the overwhelming, technological scale of the tunnel, where the stacked television sets act as solid, impassable walls."
        }
      },
      "execution": {
        "durationSeconds": 60,
        "allowFreePrompt": false,
        "maxSnapshots": 3
      },
      "skills": [
        {
          "id": "time_portal",
          "name": "🌀 Time Portal",
          "prompt": "A stable circular time portal opens between the CRT walls, showing a distinct reachable era while the mouse and tunnel remain unchanged.",
          "changeType": "one_time"
        },
        {
          "id": "channel_zero",
          "name": "Channel Zero",
          "prompt": "Every visible CRT snaps to the same luminous white eye as a wave of cyan static races down the tunnel, then the original broadcasts return.",
          "changeType": "one_time"
        },
        {
          "id": "channel_shuffle",
          "name": "🎛️ Channel Shuffle",
          "prompt": "The televisions rapidly cycle through planets, clouds, eyes, and starfields in synchronized bands moving toward the vanishing point.",
          "changeType": "one_time"
        }
      ]
    },
    "rules": {
      "skills": [
        {
          "id": "time-portal",
          "input": "🌀 Time Portal",
          "name": "🌀 Time Portal",
          "prepared": {
            "prompt": "A stable circular time portal opens between the CRT walls, showing a distinct reachable era while the mouse and tunnel remain unchanged.",
            "change_type": "one_time"
          }
        },
        {
          "id": "channel-zero",
          "input": "Channel Zero",
          "name": "Channel Zero",
          "prepared": {
            "prompt": "Every visible CRT snaps to the same luminous white eye as a wave of cyan static races down the tunnel, then the original broadcasts return.",
            "change_type": "one_time"
          }
        },
        {
          "id": "channel-shuffle",
          "input": "🎛️ Channel Shuffle",
          "name": "🎛️ Channel Shuffle",
          "prepared": {
            "prompt": "The televisions rapidly cycle through planets, clouds, eyes, and starfields in synchronized bands moving toward the vanishing point.",
            "change_type": "one_time"
          }
        }
      ]
    }
  },
  {
    "name": "Grand Prix at the End of the Stars",
    "tone": "accent",
    "size": "832x480",
    "fps": 24,
    "prompt": "A third-person trailing gameplay view keeps the clear external playable subject centered in the frame on the central vertical axis, seen from behind and facing the route ahead. The camera is set in a third-person perspective, positioned approximately five meters behind and slightly above the playable subject, who is centered in the frame on the central vertical axis. The subject is a recognizable plumber character wearing a signature red cap, blue overalls, and white gloves, seated within a compact, high-tech red racing kart. The kart features dual rear thrusters emitting bright orange flames, and the character is facing directly away from the camera, looking forward along the track. The primary traversal route is a wide, continuous, and vibrant rainbow-colored road that winds through a cosmic expanse. This track serves as the main path, extending from the foreground into the deep background, where it spirals upward into the starry void. The track is bordered by sleek, metallic guardrails that provide a clear, unobstructed path for forward movement. The environment is set in outer space, featuring a deep indigo nebula backdrop filled with glowing stars, distant planets, and streaks of colorful light from passing comets. Soft, pinkish-purple clouds drift near the track edges, adding depth to the midground. The lighting is dynamic and self-illuminated, emanating from the glowing rainbow surface of the track and the fiery exhaust of the kart, casting vivid, multi-colored reflections onto the metallic surfaces. The overall visual style is high-energy, colorful, and fantastical, emphasizing a sense of speed and wonder as the player navigates the celestial circuit.",
    "referenceUrl": `${FEISHU_INTERNAL_DEMO_ASSET_BASE_URL}/images/row-004-grand-prix-at-stars-end.jpg`,
    "mime": "image/jpeg",
    "source": "Feishu 抽卡showcase row 4",
    "worldDefinitionUrl": `${FEISHU_INTERNAL_DEMO_ASSET_BASE_URL}/worlds/row-004-grand-prix-at-stars-end.json`,
    "worldDefinition": {
      "schemaVersion": "loopit.world/v1",
      "mode": "experience",
      "id": "grand-prix-at-stars-end",
      "version": "0.1.0",
      "creationInput": "Let me race a red kart at full speed along a rainbow road spiraling through space.",
      "presentation": {
        "title": "Grand Prix at the End of the Stars",
        "introduction": "The final rainbow circuit has opened beyond the stars. Take the red kart and race where gravity cannot follow."
      },
      "world": {
        "initialSetup": {
          "perspective": "third_person",
          "startingFrame": {
            "source": "upload",
            "uri": "asset://pending"
          },
          "description": "A third-person trailing gameplay view keeps the clear external playable subject centered in the frame on the central vertical axis, seen from behind and facing the route ahead. The camera is set in a third-person perspective, positioned approximately five meters behind and slightly above the playable subject, who is centered in the frame on the central vertical axis. The subject is a recognizable plumber character wearing a signature red cap, blue overalls, and white gloves, seated within a compact, high-tech red racing kart. The kart features dual rear thrusters emitting bright orange flames, and the character is facing directly away from the camera, looking forward along the track. The primary traversal route is a wide, continuous, and vibrant rainbow-colored road that winds through a cosmic expanse. This track serves as the main path, extending from the foreground into the deep background, where it spirals upward into the starry void. The track is bordered by sleek, metallic guardrails that provide a clear, unobstructed path for forward movement. The environment is set in outer space, featuring a deep indigo nebula backdrop filled with glowing stars, distant planets, and streaks of colorful light from passing comets. Soft, pinkish-purple clouds drift near the track edges, adding depth to the midground. The lighting is dynamic and self-illuminated, emanating from the glowing rainbow surface of the track and the fiery exhaust of the kart, casting vivid, multi-colored reflections onto the metallic surfaces. The overall visual style is high-energy, colorful, and fantastical, emphasizing a sense of speed and wonder as the player navigates the celestial circuit."
        }
      },
      "execution": {
        "durationSeconds": 60,
        "allowFreePrompt": false,
        "maxSnapshots": 3
      },
      "skills": [
        {
          "id": "rocket_boost",
          "name": "🚀 Rocket Boost",
          "prompt": "The kart's rear thrusters flare blue-orange and launch it forward as stars and track lights stretch into vivid motion lines.",
          "changeType": "one_time"
        },
        {
          "id": "rainbow_drift",
          "name": "🌈 Rainbow Drift",
          "prompt": "The kart swings into a long controlled drift, painting a bright rainbow arc across the track before straightening out.",
          "changeType": "one_time"
        },
        {
          "id": "star_jump",
          "name": "⭐ Star Jump",
          "prompt": "A luminous section of track rises into a smooth ramp and launches the kart briefly into open space before it lands safely back on the rainbow road.",
          "changeType": "one_time"
        }
      ]
    },
    "rules": {
      "skills": [
        {
          "id": "rocket-boost",
          "input": "🚀 Rocket Boost",
          "name": "🚀 Rocket Boost",
          "prepared": {
            "prompt": "The kart's rear thrusters flare blue-orange and launch it forward as stars and track lights stretch into vivid motion lines.",
            "change_type": "one_time"
          }
        },
        {
          "id": "rainbow-drift",
          "input": "🌈 Rainbow Drift",
          "name": "🌈 Rainbow Drift",
          "prepared": {
            "prompt": "The kart swings into a long controlled drift, painting a bright rainbow arc across the track before straightening out.",
            "change_type": "one_time"
          }
        },
        {
          "id": "star-jump",
          "input": "⭐ Star Jump",
          "name": "⭐ Star Jump",
          "prepared": {
            "prompt": "A luminous section of track rises into a smooth ramp and launches the kart briefly into open space before it lands safely back on the rainbow road.",
            "change_type": "one_time"
          }
        }
      ]
    }
  },
  {
    "name": "The Last Pilgrimage to the Sky-Wheel",
    "tone": "accent",
    "size": "832x480",
    "fps": 24,
    "prompt": "A third-person trailing gameplay view keeps the clear external playable subject centered in the frame on the central vertical axis, seen from behind and facing the route ahead. The camera uses a third-person trailing follow perspective, positioned approximately five meters behind the playable subject. The wind surveyor is centered in the frame, occupying about 25% of the vertical height, and is seen from behind as they walk forward along a clear, dirt-packed path. The subject is a slender figure wearing a weathered, dark-brown hooded cloak that drapes over their shoulders and reaches their mid-calves, paired with sturdy, dark leather boots. The character's silhouette is distinct against the bright, expansive horizon. The traversable route is a wide, flat, and unobstructed dirt trail that cuts through a vast, rolling amber-colored grassland, leading directly toward the base of the massive Sky-Wheel Giant Tower in the distance. The terrain is covered in dry, golden-hued grasses that sway gently in the breeze. The tower itself is a monumental structure composed of dark, craggy rock, featuring a glowing, concentric circular light core at its center and three large, semi-transparent golden moon-blades that rotate slowly around the upper spire. The atmosphere is clear and bright, with soft, warm sunlight casting long, gentle shadows across the plains. The sky is a deep, vibrant blue, populated by large, fluffy, sun-drenched cumulus clouds that frame the tower. The overall mood is one of epic discovery and serene exploration, with a color palette dominated by rich ambers, deep earth tones, and brilliant golden light, creating a sense of ancient mystery and vast, open space.",
    "referenceUrl": `${FEISHU_INTERNAL_DEMO_ASSET_BASE_URL}/images/row-005-skywheel-tower.jpg`,
    "mime": "image/jpeg",
    "source": "Feishu 抽卡showcase row 5",
    "worldDefinitionUrl": `${FEISHU_INTERNAL_DEMO_ASSET_BASE_URL}/worlds/row-005-skywheel-tower.json`,
    "worldDefinition": {
      "schemaVersion": "loopit.world/v1",
      "mode": "experience",
      "id": "skywheel-tower",
      "version": "0.1.0",
      "creationInput": "Let me cross an amber grassland and follow the old road to a colossal golden Sky-Wheel tower.",
      "presentation": {
        "title": "The Last Pilgrimage to the Sky-Wheel",
        "introduction": "Beyond the amber plains, the ancient Sky-Wheel has begun to turn again. Follow the old road and find out why."
      },
      "world": {
        "initialSetup": {
          "perspective": "third_person",
          "startingFrame": {
            "source": "upload",
            "uri": "asset://pending"
          },
          "description": "A third-person trailing gameplay view keeps the clear external playable subject centered in the frame on the central vertical axis, seen from behind and facing the route ahead. The camera uses a third-person trailing follow perspective, positioned approximately five meters behind the playable subject. The wind surveyor is centered in the frame, occupying about 25% of the vertical height, and is seen from behind as they walk forward along a clear, dirt-packed path. The subject is a slender figure wearing a weathered, dark-brown hooded cloak that drapes over their shoulders and reaches their mid-calves, paired with sturdy, dark leather boots. The character's silhouette is distinct against the bright, expansive horizon. The traversable route is a wide, flat, and unobstructed dirt trail that cuts through a vast, rolling amber-colored grassland, leading directly toward the base of the massive Sky-Wheel Giant Tower in the distance. The terrain is covered in dry, golden-hued grasses that sway gently in the breeze. The tower itself is a monumental structure composed of dark, craggy rock, featuring a glowing, concentric circular light core at its center and three large, semi-transparent golden moon-blades that rotate slowly around the upper spire. The atmosphere is clear and bright, with soft, warm sunlight casting long, gentle shadows across the plains. The sky is a deep, vibrant blue, populated by large, fluffy, sun-drenched cumulus clouds that frame the tower. The overall mood is one of epic discovery and serene exploration, with a color palette dominated by rich ambers, deep earth tones, and brilliant golden light, creating a sense of ancient mystery and vast, open space."
        }
      },
      "execution": {
        "durationSeconds": 60,
        "allowFreePrompt": false,
        "maxSnapshots": 3
      },
      "skills": [
        {
          "id": "wake_wheel",
          "name": "Wake Wheel",
          "prompt": "The Sky-Wheel's golden moon-blades accelerate and its concentric core ignites with warm radiant light.",
          "changeType": "persistent"
        },
        {
          "id": "windwave",
          "name": "🌾 Windwave",
          "prompt": "A powerful wave of wind travels across the amber grassland toward the tower, bending the grass in broad flowing patterns around the traveler.",
          "changeType": "one_time"
        },
        {
          "id": "light_path",
          "name": "✨ Light Path",
          "prompt": "Ancient golden symbols illuminate one after another along the dirt trail, forming a clear glowing route to the tower's base.",
          "changeType": "persistent"
        }
      ]
    },
    "rules": {
      "skills": [
        {
          "id": "wake-wheel",
          "input": "Wake Wheel",
          "name": "Wake Wheel",
          "prepared": {
            "prompt": "The Sky-Wheel's golden moon-blades accelerate and its concentric core ignites with warm radiant light.",
            "change_type": "persistent"
          }
        },
        {
          "id": "windwave",
          "input": "🌾 Windwave",
          "name": "🌾 Windwave",
          "prepared": {
            "prompt": "A powerful wave of wind travels across the amber grassland toward the tower, bending the grass in broad flowing patterns around the traveler.",
            "change_type": "one_time"
          }
        },
        {
          "id": "light-path",
          "input": "✨ Light Path",
          "name": "✨ Light Path",
          "prepared": {
            "prompt": "Ancient golden symbols illuminate one after another along the dirt trail, forming a clear glowing route to the tower's base.",
            "change_type": "persistent"
          }
        }
      ]
    }
  },
  {
    "name": "Where the Faces Remember",
    "tone": "accent",
    "size": "832x480",
    "fps": 24,
    "prompt": "A third-person trailing gameplay view keeps the clear external playable subject centered in the frame on the central vertical axis, seen from behind and facing the route ahead. The camera uses a third-person trailing follow perspective, positioned slightly above and behind the traveler, who is centered in the frame on the central vertical axis. The traveler is a lone figure of medium height, seen from behind, wearing a weathered, dark-colored long coat and sturdy boots, standing firmly on the cracked, dry earth. The traveler's right arm is extended, with their palm resting against the cheek of a massive, weathered stone face sculpture. This sculpture, occupying the midground, is currently in the process of awakening; its eyes have opened to gaze at the traveler, and a gentle, serene smile has formed on its lips. From the point of contact, intricate golden light and stone particles are spreading across the surface, causing segments of the face to break away into fragments. The environment is a vast, desolate grey wasteland extending to the horizon, characterized by parched, fissured ground littered with gravel and weathered rock. A clear, traversable path of cracked earth winds forward between the sculptures, leading the player toward the distant, hazy horizon, inviting exploration of this surreal, silent world. Scattered across this terrain are numerous other stone face sculptures of varying sizes, some partially buried, others standing upright, all currently dormant with closed eyes. Interspersed among the sculptures are antique golden pocket watches and clock faces, some half-submerged in the dust, providing rare warm highlights against the muted, monochromatic landscape. The sky is dominated by heavy, dark storm clouds, with dim, diffused light filtering through gaps to cast long, soft shadows. The atmosphere is quiet, mysterious, and melancholic, with a visual style that emphasizes texture and scale.",
    "referenceUrl": `${FEISHU_INTERNAL_DEMO_ASSET_BASE_URL}/images/row-025-field-of-faces.jpg`,
    "mime": "image/jpeg",
    "source": "Feishu 抽卡showcase row 25",
    "worldDefinitionUrl": `${FEISHU_INTERNAL_DEMO_ASSET_BASE_URL}/worlds/row-025-field-of-faces.json`,
    "worldDefinition": {
      "schemaVersion": "loopit.world/v1",
      "mode": "experience",
      "id": "field-of-faces",
      "version": "0.1.0",
      "creationInput": "Let me wander through a field of stone faces and clocks, waking each forgotten expression with a touch.",
      "presentation": {
        "title": "Where the Faces Remember",
        "introduction": "Wander through a field of stone faces and clocks, waking each forgotten expression with a touch."
      },
      "world": {
        "initialSetup": {
          "perspective": "third_person",
          "startingFrame": {
            "source": "upload",
            "uri": "asset://pending"
          },
          "description": "A third-person trailing gameplay view keeps the clear external playable subject centered in the frame on the central vertical axis, seen from behind and facing the route ahead. The camera uses a third-person trailing follow perspective, positioned slightly above and behind the traveler, who is centered in the frame on the central vertical axis. The traveler is a lone figure of medium height, seen from behind, wearing a weathered, dark-colored long coat and sturdy boots, standing firmly on the cracked, dry earth. The traveler's right arm is extended, with their palm resting against the cheek of a massive, weathered stone face sculpture. This sculpture, occupying the midground, is currently in the process of awakening; its eyes have opened to gaze at the traveler, and a gentle, serene smile has formed on its lips. From the point of contact, intricate golden light and stone particles are spreading across the surface, causing segments of the face to break away into fragments. The environment is a vast, desolate grey wasteland extending to the horizon, characterized by parched, fissured ground littered with gravel and weathered rock. A clear, traversable path of cracked earth winds forward between the sculptures, leading the player toward the distant, hazy horizon, inviting exploration of this surreal, silent world. Scattered across this terrain are numerous other stone face sculptures of varying sizes, some partially buried, others standing upright, all currently dormant with closed eyes. Interspersed among the sculptures are antique golden pocket watches and clock faces, some half-submerged in the dust, providing rare warm highlights against the muted, monochromatic landscape. The sky is dominated by heavy, dark storm clouds, with dim, diffused light filtering through gaps to cast long, soft shadows. The atmosphere is quiet, mysterious, and melancholic, with a visual style that emphasizes texture and scale."
        }
      },
      "execution": {
        "durationSeconds": 60,
        "allowFreePrompt": false,
        "maxSnapshots": 3
      },
      "skills": [
        {
          "id": "awaken_face",
          "name": "Awaken",
          "prompt": "Touching a stone face makes it open its eyes and smile before crumbling into a pile of rock at the same location.",
          "changeType": "one_time"
        },
        {
          "id": "stone_prayer",
          "name": "🙏 Prayer",
          "prompt": "The traveler kneels in prayer and every stone face across the field opens its eyes at once.",
          "changeType": "one_time"
        },
        {
          "id": "rewind_clock",
          "name": "⏪ Rewind Clock",
          "prompt": "Every scattered clock spins backward as cracks close across the nearest stone face.",
          "changeType": "one_time"
        }
      ]
    },
    "rules": {
      "skills": [
        {
          "id": "awaken-face",
          "input": "Awaken",
          "name": "Awaken",
          "prepared": {
            "prompt": "Touching a stone face makes it open its eyes and smile before crumbling into a pile of rock at the same location.",
            "change_type": "one_time"
          }
        },
        {
          "id": "stone-prayer",
          "input": "🙏 Prayer",
          "name": "🙏 Prayer",
          "prepared": {
            "prompt": "The traveler kneels in prayer and every stone face across the field opens its eyes at once.",
            "change_type": "one_time"
          }
        },
        {
          "id": "rewind-clock",
          "input": "⏪ Rewind Clock",
          "name": "⏪ Rewind Clock",
          "prepared": {
            "prompt": "Every scattered clock spins backward as cracks close across the nearest stone face.",
            "change_type": "one_time"
          }
        }
      ]
    }
  },
  {
    "name": "The Queen Beneath the Snow",
    "tone": "accent",
    "size": "832x480",
    "fps": 24,
    "prompt": "Third-person trailing view. The player controls an arctic traveler positioned near the center of the frame, seen from behind, walking through a wide, level, continuous snow basin. The snowfield route passes low ice ridges and leads toward a colossal female statue head half-buried in the distant snow. Her face is carved from pale gray-white rock, while waterfall-like ice resembling long hair covers her forehead and both sides of her face; only the upper half of her face and silhouette emerge from the snow.\nSnow-covered mountain ridges and sparse rocks line both sides, while a blizzard continues to fall. The character, route, and statue head remain clearly visible at all times. Over approximately one minute, the traveler approaches the statue across the snowfield. Avoid additional characters, vehicles, weapons, modern buildings, narrow bridges, text, and UI elements.",
    "referenceUrl": `${FEISHU_INTERNAL_DEMO_ASSET_BASE_URL}/images/row-029-queen-beneath-snow.jpg`,
    "mime": "image/jpeg",
    "source": "Feishu 抽卡showcase row 29",
    "worldDefinitionUrl": `${FEISHU_INTERNAL_DEMO_ASSET_BASE_URL}/worlds/row-029-queen-beneath-snow.json`,
    "worldDefinition": {
      "schemaVersion": "loopit.world/v1",
      "mode": "experience",
      "id": "queen-beneath-snow",
      "version": "0.1.0",
      "creationInput": "Let me cross a blizzard-swept basin toward a colossal goddess head buried in the ice.",
      "presentation": {
        "title": "The Queen Beneath the Snow",
        "introduction": "An ancient face sleeps beneath the frozen mountains. Cross the white basin and discover what the storm has kept hidden."
      },
      "world": {
        "initialSetup": {
          "perspective": "third_person",
          "startingFrame": {
            "source": "upload",
            "uri": "asset://pending"
          },
          "description": "Third-person trailing view. The player controls an arctic traveler positioned near the center of the frame, seen from behind, walking through a wide, level, continuous snow basin. The snowfield route passes low ice ridges and leads toward a colossal female statue head half-buried in the distant snow. Her face is carved from pale gray-white rock, while waterfall-like ice resembling long hair covers her forehead and both sides of her face; only the upper half of her face and silhouette emerge from the snow.\nSnow-covered mountain ridges and sparse rocks line both sides, while a blizzard continues to fall. The character, route, and statue head remain clearly visible at all times. Over approximately one minute, the traveler approaches the statue across the snowfield. Avoid additional characters, vehicles, weapons, modern buildings, narrow bridges, text, and UI elements."
        }
      },
      "execution": {
        "durationSeconds": 60,
        "allowFreePrompt": false,
        "maxSnapshots": 3
      },
      "skills": [
        {
          "id": "whiteout",
          "name": "❄️ Whiteout",
          "prompt": "The snowfall intensifies into a dense blizzard while the traveler and buried face remain readable.",
          "changeType": "persistent"
        },
        {
          "id": "open_eyes",
          "name": "👁️ Open Eyes",
          "prompt": "The colossal buried goddess slowly opens her eyes and releases a faint icy glow.",
          "changeType": "one_time"
        },
        {
          "id": "ice_breath",
          "name": "🌬️ Ice Breath",
          "prompt": "The goddess exhales a pale current that parts the blizzard and reveals a clear route across the basin.",
          "changeType": "one_time"
        }
      ]
    },
    "rules": {
      "skills": [
        {
          "id": "whiteout",
          "input": "❄️ Whiteout",
          "name": "❄️ Whiteout",
          "prepared": {
            "prompt": "The snowfall intensifies into a dense blizzard while the traveler and buried face remain readable.",
            "change_type": "persistent"
          }
        },
        {
          "id": "open-eyes",
          "input": "👁️ Open Eyes",
          "name": "👁️ Open Eyes",
          "prepared": {
            "prompt": "The colossal buried goddess slowly opens her eyes and releases a faint icy glow.",
            "change_type": "one_time"
          }
        },
        {
          "id": "ice-breath",
          "input": "🌬️ Ice Breath",
          "name": "🌬️ Ice Breath",
          "prepared": {
            "prompt": "The goddess exhales a pale current that parts the blizzard and reveals a clear route across the basin.",
            "change_type": "one_time"
          }
        }
      ]
    }
  }
];
})(globalThis);
